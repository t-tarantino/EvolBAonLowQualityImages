"""
config.py — single source of truth for all hyperparameters and constants.
Change values here; every other module picks them up automatically.
"""

# ── Reproducibility ──────────────────────────────────────────────────────────
SEED = 42

# ── EvolBA hyperparameters ───────────────────────────────────────────────────
# Phase 2 – binary search
BS_STEPS = 15           # number of bisection steps; precision ≈ 1/2^BS_STEPS

# Phase 3 – CMA-ES
PENALTY  = 20.0         # fitness penalty for clean (non-adversarial) candidates;
                        # must exceed any realistic ||x_adv - x||_2

# Query budgets (can be overridden per call)
MAX_Q_FMNIST  = 2000    # Fashion-MNIST (784 dims)
MAX_Q_CIFAR10 = 3000    # CIFAR-10 (3072 dims — harder, needs more queries)

# ── Data ─────────────────────────────────────────────────────────────────────
DATA_ROOT  = './data'
N_COLLECT  = 200        # pool of correctly-classified images to draw attacks from
N_EVAL     = 50         # how many to actually attack in a batch run

# ── Fashion-MNIST ─────────────────────────────────────────────────────────────
FMNIST_WEIGHTS = 'cnn_fashionmnist.pth'
FMNIST_EPOCHS  = 10
FMNIST_CLASSES = [
    'T-shirt', 'Trouser', 'Pullover', 'Dress', 'Coat',
    'Sandal',  'Shirt',   'Sneaker',  'Bag',   'Ankle boot',
]

# ── CIFAR-10 ──────────────────────────────────────────────────────────────────
# Per-channel mean and std used during ResNet-56 training (chenyaofo repo)
CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD  = (0.2023, 0.1994, 0.2010)
CIFAR10_HUB  = 'chenyaofo/pytorch-cifar-models'
CIFAR10_MODEL = 'cifar10_resnet56'   # ~93.4% test accuracy

# RobustBench — adversarially trained model (L∞, ε=8/255)
CIFAR10_ROBUST_MODEL  = 'Wang2023Better_WRN-28-10'  # ~92.4% clean, ~67.3% robust
CIFAR10_ROBUST_THREAT = 'Linf'

CIFAR10_CLASSES = [
    'airplane', 'automobile', 'bird', 'cat',  'deer',
    'dog',      'frog',       'horse', 'ship', 'truck',
]

# ── attacks/ framework — Phase 1 ─────────────────────────────────────────────
P1_GAUSSIAN_MAX_ATTEMPTS = 500     # GaussianInit: noise escalation steps
P1_GAUSSIAN_SCALE_MIN    = 0.05    # GaussianInit: starting noise std
P1_GAUSSIAN_SCALE_MAX    = 2.0     # GaussianInit: max noise std
P1_FAMILY_BS_STEPS       = 12      # CorruptionFamilyInit: binary search steps

# ── attacks/ framework — Phase 3 general ────────────────────────────────────
P3_SNAP_FRACS = (0.25, 0.50, 0.75)  # budget fractions at which to snapshot

# ── attacks/ framework — sep-CMA-ES ─────────────────────────────────────────
SEP_CMAES_LAM    = None    # None → 4 + floor(3·ln n) (auto)
SEP_CMAES_SIGMA0 = None    # None → 0.1 · dist_init / √n (auto)

# ── attacks/ framework — (1+1)-CMA-ES ───────────────────────────────────────
OPO_P_TARGET  = 0.2    # target success rate for 1/5 rule
OPO_WINDOW    = 10     # rolling window for success rate estimate
OPO_SIGMA_INC = 1.22   # sigma multiplier on success
OPO_SIGMA_DEC = 0.82   # sigma multiplier on failure

# ── attacks/ framework — subspace ───────────────────────────────────────────
SUBSPACE_K_DCT = 100   # DCT components in combined basis
SUBSPACE_K_SP  = 32    # grid-superpixel components in combined basis
