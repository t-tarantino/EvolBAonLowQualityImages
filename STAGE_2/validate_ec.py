#!/usr/bin/env python3
"""
validate_ec.py — sanity checks for evolba_ec.py before running ec1.

1. Unit tests for the VkD linear algebra (_vkd_sample, _vkd_update) on
   synthetic data, no oracle needed.
2. Regression test: evolba_vkd(vk_rank=0) must reproduce evolba_tuned()
   EXACTLY (same trajectory, same everything) given identical params/seed.
3. Smoke test: evolba_vkd(vk_rank=1,2,3) and evolba_one_plus_one() run to
   completion on a real CIFAR-10 image/model without errors, with sane
   diagnostics (V grows from zero, (1+1) is elitist/monotonic).

Usage: python validate_ec.py
"""
import os, sys, warnings
import numpy as np
warnings.filterwarnings('ignore')

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'STAGE_1'))
sys.path.insert(0, os.path.dirname(__file__))

from evolba_ec import evolba_vkd, evolba_one_plus_one, _vkd_sample, _vkd_update

PASS, FAIL = [], []
def check(name, cond, detail=''):
    if cond:
        PASS.append(name)
        print(f'  [PASS] {name}')
    else:
        FAIL.append(name)
        print(f'  [FAIL] {name}  {detail}')


# ════════════════════════════════════════════════════════════════════════════
print('=== 1. Unit tests: _vkd_sample / _vkd_update (synthetic, no oracle) ===')

n = 5

# -- _vkd_sample: V = s * e0 (rank-1, aligned with axis 0) --------------------
s = 2.0
V = np.zeros((n, 1)); V[0, 0] = s
zs = np.array([
    [1.0, 0, 0, 0, 0],   # aligned with V's direction
    [0.0, 1, 0, 0, 0],   # orthogonal to V's direction
])
ys = _vkd_sample(zs, V)
expected_scale = np.sqrt(1.0 + s ** 2)
check('_vkd_sample: aligned component scaled by sqrt(1+s^2)',
      np.isclose(abs(ys[0, 0]), expected_scale, atol=1e-8),
      f'got {ys[0,0]}, expected ±{expected_scale}')
check('_vkd_sample: orthogonal component unchanged',
      np.allclose(ys[1], zs[1], atol=1e-8),
      f'got {ys[1]}')

# k=0 must be a no-op (identity), regardless of zs
V0 = np.zeros((n, 0))
zs_rand = np.random.default_rng(0).standard_normal((4, n))
check('_vkd_sample: vk_rank=0 is identity',
      np.array_equal(_vkd_sample(zs_rand, V0), zs_rand))

# -- _vkd_update: offspring with most weighted variance along e0 -------------
V = np.zeros((n, 1))
zs_ranked = np.array([
    [3.0, 0, 0, 0, 0],
    [2.0, 0, 0, 0, 0],
    [1.0, 0, 0, 0, 0],
])
w_eff = np.array([0.5, 0.3, 0.2])

Y = zs_ranked * w_eff[:, None]              # [[1.5,0,0,0,0],[0.6,0,0,0,0],[0.2,0,0,0,0]]
S_y = np.linalg.svd(Y, compute_uv=False)    # length 3, only S_y[0] nonzero
eig = S_y ** 2
baseline = eig.mean()
expected_excess = max(0.0, eig[0] - baseline)
expected_norm = np.sqrt(expected_excess / baseline)

V_new = _vkd_update(V, zs_ranked, w_eff, cv=1.0)
check('_vkd_update: column 0 aligns with e0 axis',
      np.allclose(V_new[1:, 0], 0.0, atol=1e-8),
      f'got {V_new[:,0]}')
check('_vkd_update: column 0 magnitude matches sqrt(excess/baseline) prediction',
      np.isclose(abs(V_new[0, 0]), expected_norm, atol=1e-6),
      f'got {abs(V_new[0,0])}, expected {expected_norm}')

# k=0 must be a no-op
check('_vkd_update: vk_rank=0 is no-op',
      _vkd_update(np.zeros((n, 0)), zs_ranked, w_eff, cv=1.0).shape == (n, 0))


# ════════════════════════════════════════════════════════════════════════════
print('\n=== Loading CIFAR-10 + standard model (for tests 2 & 3) ===')
import torch
import torchvision
from robustbench.utils import load_model

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'  device: {device}')

model = load_model('Standard', dataset='cifar10', threat_model='Linf').to(device).eval()
def oracle(x_chw):
    with torch.no_grad():
        t = torch.from_numpy(x_chw[None].astype(np.float32)).to(device)
        return int(model(t).argmax(1).item())

ds = torchvision.datasets.CIFAR10(root='/tmp/cifar10', train=False, download=True)
x_orig, y_true = None, None
for img_pil, label in ds:
    x = np.array(img_pil, dtype=np.float32).transpose(2, 0, 1) / 255.0
    if oracle(x) == label:
        x_orig, y_true = x, label
        break
print(f'  test image: true_class={y_true}')

MAX_Q = 300
CARRIER = dict(xi_step_scale=0.5, bs_adaptive=True, cmu_scale=1.0,
               lam_override=None, tau=3, max_queries=MAX_Q, seed=42)


# ════════════════════════════════════════════════════════════════════════════
print('\n=== 2. Regression test: evolba_vkd(vk_rank=0) == evolba_tuned ===')
from evolba_tuned import evolba_tuned

r_tuned = evolba_tuned(oracle, x_orig, y_true, **CARRIER)
r_vk0   = evolba_vkd(oracle, x_orig, y_true, vk_rank=0, **CARRIER)

check('queries match',       r_tuned['queries']       == r_vk0['queries'])
check('n_generations match', r_tuned['n_generations'] == r_vk0['n_generations'])
check('final_l2 match',       np.isclose(r_tuned['final_l2'], r_vk0['final_l2']))
check('best_l2 match',         np.isclose(r_tuned['best_l2'],  r_vk0['best_l2']))
check('init_l2 match',         np.isclose(r_tuned['init_l2'],  r_vk0['init_l2']))
check('shrink_iters match',  r_tuned['shrink_iters']  == r_vk0['shrink_iters'])
check('backtracks match',    r_tuned['backtracks']    == r_vk0['backtracks'])
check('trajectory identical',
      r_tuned['trajectory'] == r_vk0['trajectory'],
      f"tuned[:3]={r_tuned['trajectory'][:3]} vk0[:3]={r_vk0['trajectory'][:3]}")


# ════════════════════════════════════════════════════════════════════════════
print('\n=== 3. Smoke test: VkD k=1,2,3 ===')
for k in (1, 2, 3):
    r = evolba_vkd(oracle, x_orig, y_true, vk_rank=k, **CARRIER)
    v_final = r['v_norms'][-1] if r['v_norms'] else []
    v_max_over_run = np.max(r['v_norms']) if r['v_norms'] and len(r['v_norms'][0]) else 0.0
    check(f'k={k}: completes, finite final_l2',
          np.isfinite(r['final_l2']),
          f"final_l2={r['final_l2']}")
    check(f'k={k}: n_generations > 0', r['n_generations'] > 0)
    check(f'k={k}: V grows from zero (max column norm over run > 1e-6)',
          v_max_over_run > 1e-6,
          f'v_norms[-1]={v_final}')
    print(f'    k={k}: n_gen={r["n_generations"]}, final_l2={r["final_l2"]:.4f}, '
          f'best_l2={r["best_l2"]:.4f}, v_norms[-1]={v_final}')


# ════════════════════════════════════════════════════════════════════════════
print('\n=== 4. Smoke test: evolba_one_plus_one ===')
r11 = evolba_one_plus_one(oracle, x_orig, y_true, bs_adaptive=True,
                           max_queries=MAX_Q, seed=42)
l2s = [l2 for _, l2 in r11['trajectory']]
monotonic = all(l2s[i+1] <= l2s[i] + 1e-9 for i in range(len(l2s)-1))
check('(1+1): completes, finite final_l2', np.isfinite(r11['final_l2']))
check('(1+1): n_generations > 0', r11['n_generations'] > 0)
check('(1+1): best_l2 == final_l2 (elitist)',
      np.isclose(r11['best_l2'], r11['final_l2']))
check('(1+1): trajectory L2 is monotonically non-increasing',
      monotonic, f'l2s={l2s[:10]}...')
check('(1+1): sigma trajectory finite and positive',
      all(np.isfinite(r11['sigma_trajectory'])) and all(s > 0 for s in r11['sigma_trajectory']))
print(f"    n_gen={r11['n_generations']}, n_successes={r11['n_successes']}, "
      f"final_l2={r11['final_l2']:.4f}, init_l2={r11['init_l2']:.4f}, "
      f"sigma[0]={r11['sigma_trajectory'][0]:.6f}, sigma[-1]={r11['sigma_trajectory'][-1]:.6f}")


# ════════════════════════════════════════════════════════════════════════════
print(f'\n=== Summary: {len(PASS)} passed, {len(FAIL)} failed ===')
if FAIL:
    print('FAILED:')
    for f in FAIL:
        print(f'  - {f}')
    sys.exit(1)
print('All checks passed.')
