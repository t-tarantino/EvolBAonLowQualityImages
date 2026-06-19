#!/usr/bin/env python3
"""
diag_bs_adaptive.py — diagnostic: does binary_search's float32 saturation
ever actually trigger within a realistic step budget, and if so, when?

For a handful of real (x_adv, x_orig, y_true) pairs collected from real
evolba_tuned trajectories at Q=2000, run an EXTENDED bisection (n_steps=40,
well past the cap=26 used in Study 5) and report, for each pair:
  - L = ||x_adv - x_orig||  (segment length)
  - n_true: the step at which mid first equals lo or hi (np.array_equal),
            i.e. the TRUE saturation point, independent of any cap
  - whether n_true <= 26 (would Study 5's cap=26 early-stop have fired?)

Also dumps per-step bracket info for a couple of representative pairs
(one early-trajectory / large L, one late-trajectory / small L) so we can
see exactly what "mid == lo" looks like in float32.
"""
import os, sys, warnings
import numpy as np
import torch

warnings.filterwarnings('ignore')
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.dirname(__file__))

from evolba_tuned import evolba_tuned

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'Device: {device}')

from robustbench.utils import load_model
model_std = load_model('Standard', dataset='cifar10', threat_model='Linf').to(device).eval()
model_rob = load_model('Wang2023Better_WRN-28-10', dataset='cifar10', threat_model='Linf').to(device).eval()

def make_oracle(model):
    def oracle(x_chw):
        with torch.no_grad():
            t = torch.from_numpy(x_chw[None].astype(np.float32)).to(device)
            return int(model(t).argmax(1).item())
    return oracle

MODELS = {'standard': make_oracle(model_std), 'robust': make_oracle(model_rob)}

import torchvision
ds = torchvision.datasets.CIFAR10(root='/tmp/cifar10', train=False, download=True)
images, labels = [], []
counts = [0]*10
for img_pil, label in ds:
    if counts[label] >= 1: continue
    x = np.array(img_pil, dtype=np.float32).transpose(2,0,1)/255.0
    if MODELS['standard'](x) == label:
        images.append(x); labels.append(label); counts[label] += 1
    if sum(counts) == 3: break  # 3 images is plenty for a diagnostic

HPARAMS = dict(xi_step_scale=0.5, tau=1, lam_override=14, cmu_scale=1.0)
MAX_Q   = 2000

def manual_bisect(x_adv, x_orig, n_steps=40, verbose_pair=False):
    """Run bisection WITHOUT an oracle -- just track bracket geometry and find
    the first step where mid == lo or mid == hi (float32). The actual
    direction (lo=mid vs hi=mid) doesn't affect *when* saturation occurs
    (the bracket halves each step regardless of which side moves), so we
    alternate deterministically (always move hi->mid) which still produces
    the same halving sequence for |hi-lo|.
    """
    lo, hi = x_orig.copy(), x_adv.copy()
    for k in range(1, n_steps+1):
        mid = np.clip(0.5*(lo+hi), 0.0, 1.0).astype(np.float32)
        eq_lo = np.array_equal(mid, lo)
        eq_hi = np.array_equal(mid, hi)
        if verbose_pair and k <= 30:
            bracket_norm = float(np.linalg.norm((hi.astype(np.float64)-lo.astype(np.float64)).flatten()))
            print(f'    step {k:2d}: ||hi-lo||={bracket_norm:.3e}  mid==lo:{eq_lo}  mid==hi:{eq_hi}')
        if eq_lo or eq_hi:
            return k - 1   # n_true = number of *useful* steps before saturation
        # alternate which side moves -- doesn't change the halving rate
        hi = mid
    return n_steps

print('\n=== Collecting (x_adv, x_orig, y_true) pairs from real trajectories (Q=2000) ===')
all_pairs = []  # (model, gen_idx, dist_to_orig_proxy, x_adv, x_orig, y_true)

for mname, oracle in MODELS.items():
    for img_idx in range(3):
        x_orig = images[img_idx]
        y_true = int(labels[img_idx])
        pairs  = []
        result = evolba_tuned(oracle, x_orig, y_true, max_queries=MAX_Q, seed=img_idx*100,
                              bs_adaptive=True, bs_cap=26, collect_bs_pairs=pairs, **HPARAMS)
        print(f'{mname} img{img_idx}: init_l2={result["init_l2"]:.3f} best_l2={result["best_l2"]:.3f} '
              f'gen={result["n_generations"]} bs_calls={result["bs_calls"]} '
              f'pairs_collected={len(pairs)}')
        for gi, (x_adv, x_orig_p, yt) in enumerate(pairs):
            all_pairs.append((mname, gi, x_adv, x_orig_p, yt))

print(f'\nTotal pairs collected: {len(all_pairs)}')

print('\n=== Extended (n_steps=40) saturation check on a sample ===')
import random
random.seed(0)
sample = random.sample(all_pairs, min(40, len(all_pairs)))
results = []
for mname, gi, x_adv, x_orig_p, yt in sample:
    L = float(np.linalg.norm((x_adv.astype(np.float64)-x_orig_p.astype(np.float64)).flatten()))
    n_true = manual_bisect(x_adv, x_orig_p, n_steps=40)
    results.append((mname, gi, L, n_true))
    print(f'  {mname:<8s} gen={gi:3d}  L={L:7.4f}  n_true={n_true:3d}  '
          f'(<=26? {"YES" if n_true<=26 else "no"})')

n_le_26 = sum(1 for *_, n in results if n <= 26)
print(f'\n{n_le_26}/{len(results)} pairs saturate at or before step 26')
print(f'n_true range: {min(r[3] for r in results)} - {max(r[3] for r in results)}')

print('\n=== Step-by-step trace for two representative pairs ===')
# pick the pair with the largest L and the one with the smallest L
sample_sorted = sorted(results, key=lambda r: r[2])
small_L = sample_sorted[0]
large_L = sample_sorted[-1]
for label, (mname, gi, L, n_true) in [('SMALLEST L', small_L), ('LARGEST L', large_L)]:
    print(f'\n--- {label}: {mname} gen={gi} L={L:.4f} n_true={n_true} ---')
    # find the matching pair again
    for m2, g2, x_adv, x_orig_p, yt in all_pairs:
        if m2 == mname and g2 == gi:
            manual_bisect(x_adv, x_orig_p, n_steps=30, verbose_pair=True)
            break

print('\nDone.')
