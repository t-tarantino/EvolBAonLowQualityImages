# Evolutionary Black-Box Adversarial Attacks with CMA-ES

**Final report: [`report/report.pdf`](report/report.pdf).** For a map of which script produced
which result, see [`CodeStructure.md`](CodeStructure.md).

## 0. Setup

```bash
git clone https://github.com/t-tarantino/EvolBAonLowQualityImages.git
cd EvolBAonLowQualityImages
python3 -m venv EV_env
source EV_env/bin/activate          # Windows: EV_env\Scripts\activate
pip install -r requirements.txt
```

Two large assets are downloaded on first use rather than committed to the repo:
- **CIFAR-10 / FashionMNIST** — fetched automatically by `torchvision.datasets` into `data/`
  the first time any script runs.
- **RobustBench checkpoints** (`Standard`, `Wang2023Better_WRN-28-10`) — fetched automatically
  by `robustbench.utils.load_model` into `models/cifar10/Linf/` the first time any script
  loads a model.

Every `STAGE_X/`, `FINAL_RUN/`, and `report/` folder has a `models -> ../models` symlink
(already tracked in the repo) so the checkpoints only need to be downloaded once at the
top level. Scripts are run with their own folder as the working directory, e.g.:

```bash
cd STAGE_1 && python run_study6.py --mock   # quick smoke test
cd STAGE_1 && python run_study6.py          # full run
```

---

## 1. Overview

This project studies the use of **Evolutionary Strategies (ES)**—in particular **CMA-ES (Covariance Matrix Adaptation Evolution Strategy)**—to generate adversarial examples in a **hard-label black-box (HL-BB)** setting.

In this setting, the attacker:
- has **no access to model internals** (weights, gradients, architecture)
- receives only the **final predicted label** for each query

**The central goal is to minimize the number of model queries needed to produce a good-enough adversarial example** — one that fools the classifier while remaining imperceptible to a human observer. We study this under two attack scenarios:
- **Untargeted**: find any perturbation such that `f(x + δ) ≠ y`
- **Targeted**: find a perturbation such that `f(x + δ) = y_t`, where `y_t` is a specific wrong label chosen by the attacker

We analyze how CMA-ES behaves under these constraints and whether its internal mechanisms (covariance adaptation, step-size control) remain effective when the optimization signal is extremely weak.

---

## 2. Problem Definition

Given:
- a trained classifier `f(x)`
- an input image `x` with true label `y`

**Untargeted attack** — find `δ` such that:
- `f(x + δ) ≠ y`

**Targeted attack** — find `δ` such that:
- `f(x + δ) = y_t`, where `y_t ≠ y` is chosen by the attacker

In both cases, subject to:
- `||δ||` is small (imperceptibility constraint)
- only the predicted label of `f(x + δ)` is observable per query (HL-BB)

---

## 3. Goals

### Primary goal
- **Minimize the number of queries** required to produce a perceptually imperceptible adversarial example, in both untargeted and targeted settings

### Core focus — Evolutionary Optimization
This is fundamentally an **evolutionary computing project**. The central contribution is the study and improvement of CMA-ES as an evolutionary optimizer operating under the extreme constraint of binary (hard-label) feedback. Key aspects under investigation:
- which CMA-ES variant and configuration is best suited to this setting
- how each evolutionary mechanism (covariance adaptation, step-size control, selection, recombination) contributes under weak signals
- what algorithmic improvements to EvolBA's evolutionary loop reduce query cost

### Secondary goals
- Understand the per-phase query budget structure of EvolBA and which phase is the bottleneck
- Study both untargeted and targeted attack scenarios
- Compare CMA-ES against evolutionary and non-evolutionary baselines

### Research questions
- Which improvements to EvolBA's evolutionary core (CMA-ES variant, hyperparameters, fitness shaping, restart strategies) most reduce query count?
- How many queries does each phase of EvolBA consume, and which phase drives the most variance?
- How does targeted vs untargeted attack difficulty compare in terms of query cost and evolutionary dynamics?
- Does covariance adaptation provide meaningful benefit when the only feedback is a binary label?
- How does initialization strategy affect convergence speed and total query count?

---

## 4. Victim Models

### Dual-model setup: Adversarially trained vs. standard

We use **WideResNet-28-10** in two configurations to isolate the effect of training procedure on model robustness:

| Model | Training | Source | Clean Acc | L∞ Robust Acc | Speed |
|-------|----------|--------|-----------|--------------|-------|
| **RobustBench WRN-28-10** | TRADES adversarial (L∞) | RobustBench | ~92.4% | ~67.3% | Fast |
| **PyTorch standard WRN-28-10** | Standard CE loss | torch.hub | ~95%+ | ~0% | Fast |

#### Why WideResNet-28-10?
- **Optimal for CIFAR-10**: 28 layers × width-10 provides sufficient capacity (36M params) for high clean accuracy while remaining computationally efficient
- **Proven baseline**: standard architecture in adversarial robustness literature
- **Inference speed**: fast enough for large-scale corruption studies and attack experiments

#### Why dual models?
- **Controlled comparison**: identical architecture eliminates confounding factors — differences in robustness come purely from training procedure
- **Training effect isolation**: reveals which corruptions adversarial training addresses vs. which it ignores
- **Expected finding**: standard model will show sharp vulnerability to Gaussian noise and high-frequency perturbations (exploited by TRADES), while robustly trained model exhibits more uniform tolerance across corruption types

#### Robustness properties
- **RobustBench model**: empirically robust against L∞ perturbations via TRADES training; shows smooth decision boundaries
- **Standard model**: vulnerable to adversarial perturbations; exhibits sharp decision boundaries exploited by perturbations matching training data statistics

---

## 5. Dataset

### Main dataset
- **CIFAR-10**
  - 32×32 RGB color images
  - 10 classes (real-world objects)
  - chosen because color images are more representative of real adversarial settings and computation time is not a bottleneck

### Possible extensions
- **CIFAR-100** — same resolution, finer-grained classes, more challenging targeted attacks
- **ImageNet** (subset) — higher resolution, closer to production-level threat models
- **CelebA** — large-scale face attribute dataset; particularly interesting because face recognition systems are a realistic real-world deployment target (e.g. automated identification systems used in law enforcement), making adversarial attacks on faces a scenario with direct practical relevance

### Preprocessing
- Normalize pixel values to `[0, 1]`
- Train a baseline classifier (CNN / ResNet)

---

## 6. Threat Model

We assume a **strict black-box setting**:

- Access:
  - Input → predicted label only
- No access to:
  - probabilities / confidence scores
  - gradients
  - training data

This corresponds to a **hard-label black-box (HL-BB)** scenario.

---

## 7. Methods

### 7.1 CMA-ES (main method)

We use CMA-ES to optimize perturbations:

- Representation:
  - `δ ∈ ℝⁿ` (image-shaped perturbation)
- Sampling:
  - `δ_i ~ N(m, σ²C)`
- Evaluation:
  - apply perturbation: `x' = x + δ_i`
  - query model: `f(x')`

---

### 7.2 Fitness Function and Perceptual Similarity

The fitness function in EvolBA encodes **image similarity**: it rewards perturbations that are small enough to be imperceptible to humans while still crossing the decision boundary. Defining "imperceptible" is non-trivial — the goal is to produce noise that humans cannot detect but that fools AI models.

#### Candidate similarity metrics

| Metric | Description | Notes |
|---|---|---|
| **L2** | Euclidean distance in pixel space | Simple, differentiable, does not model human vision |
| **L∞** | Maximum pixel deviation | Bounds worst-case pixel change, common in literature |
| **SSIM** | Structural Similarity Index | Captures luminance, contrast, structure — closer to human perception |
| **MS-SSIM** | Multi-scale SSIM | More robust across resolutions |
| **LPIPS** | Learned Perceptual Image Patch Similarity | Deep-feature-based, best correlation with human judgements |
| **PSNR** | Peak Signal-to-Noise Ratio | Log-scale L2, used in compression literature |

#### Role of similarity in the overall problem

In the big picture, perceptual similarity is a **constraint**, not an optimization target: we want an adversarial example that satisfies `similarity(x, x+δ) ≥ threshold` (imperceptibility) while the actual optimization drives misclassification. EvolBA internalizes this by using a scalar similarity score as its fitness, guiding the search toward the decision boundary while keeping the perturbation small.

#### Why not multi-objective / Pareto optimization across metrics?

Given that there are multiple candidate metrics for perceptual similarity, one might consider running Pareto-front optimization across several of them simultaneously. We explicitly reject this approach for two reasons:

1. **Query cost**: multi-objective evolutionary algorithms require significantly more function evaluations to maintain a diverse Pareto front — this directly conflicts with our primary goal of minimizing queries.
2. **The problem is not intrinsically multi-objective**: we do not want a diverse set of trade-off images; we want a single good-enough adversarial example that satisfies the imperceptibility constraint. Diversity along the Pareto front provides no value here. The right approach is to select one metric (or a fixed weighted combination of a few) and optimize it as a **scalar fitness** with a single query stream.

#### Strategy used

- Select one primary similarity metric (candidate: SSIM or LPIPS) or a fixed weighted combination
- **Lexicographic priority**: misclassification is mandatory; among feasible candidates, minimize the chosen metric
- Among successful candidates on the boundary, select the one closest to the original image under the chosen metric

---

### 7.3 Baselines

We compare against:

- **Random search**
- **Simple Evolution Strategy (fixed Gaussian)**
- *(Optional)* **Genetic Algorithm (GA)**

---

### 7.4 Variants (Core Contributions)

#### 1. Initialization strategies
- random Gaussian noise
- structured perturbations (e.g. low-frequency or patch-based)

#### 2. Step-size adaptation
- standard CMA-ES
- modified adaptation rules

#### 3. Covariance adaptation ablation
- full covariance matrix
- diagonal covariance
- no covariance adaptation

---

## 8. Evaluation Metrics

- **Attack success rate**
- **Number of queries required**
- **Perturbation magnitude**
  - `L2`, `L∞`
- **Convergence speed**

---

## 9. Experimental Setup

- Train a classifier on the dataset
- Select correctly classified samples
- Run attacks with:
  - fixed query budget
  - multiple random seeds
- Compare methods and variants

---

## 10. Expected Results

We expect:

- CMA-ES to outperform random search
- performance degradation in HL-BB vs soft-label setting
- structured initialization to improve convergence
- covariance adaptation to be less effective under weak signals

---

## 11. Project Structure

See [`CodeStructure.md`](CodeStructure.md) for the full folder-by-folder map, including
which script produced which figure/table in the report.

---

## 12. TODO / Open Research Questions

### Perceptual Similarity — Constraint Calibration
- [ ] Since similarity is a **constraint** in the overall problem (not an optimization target), calibrate what numerical threshold for each metric corresponds to "imperceptible to humans" on CIFAR-10. Candidate approach: use existing perceptual similarity datasets (e.g. BAPPS / TID2013) or run a small human-judgement study. Key question: at what SSIM / LPIPS / L2 value does a perturbation become noticeable?
- [ ] Are the perceptibility thresholds content-dependent (e.g. textured regions vs uniform backgrounds tolerate different noise levels)? If so, should the constraint be adaptive per image?
- [ ] Empirically compare L2, L∞, SSIM, MS-SSIM, and LPIPS as scalar fitness signals within EvolBA on CIFAR-10: which metric, when used as the optimization objective, produces results that best satisfy the imperceptibility constraint at the same query budget?
- [ ] Does the choice of fitness metric affect convergence speed (queries to first misclassification), or only the quality of the final perturbation?

### Query Budget Analysis
- [ ] Profile query usage across all phases of EvolBA: how many queries does each phase consume on average over a representative CIFAR-10 sample?
- [ ] Which phase is responsible for the highest variance in total query count across different images and random seeds?
- [ ] Define sensible query budget tiers (e.g. 1k / 5k / 10k) and benchmark attack success rate vs budget for each phase independently.

### Phase 1 — Boundary Initialization (Noise Schedule)
- [ ] Is a linear increase of noise magnitude efficient for reaching the decision boundary, or does an exponential/cosine schedule get there in fewer queries?
- [ ] Should the noise schedule study be conducted jointly with Phase 2 (binary search), given that Phase 1 quality directly determines the starting point for Phase 2?
- [ ] How sensitive are Phase 2 and Phase 3 to the quality of the boundary estimate produced by Phase 1?

### Phase 1 & 2 Tradeoff
- [ ] How much does improving Phase 1 boundary quality reduce Phase 2 binary search depth, and vice versa?
- [ ] Is there a sweet spot in binary search iterations (Phase 2) beyond which further refinement yields no measurable gain in Phase 3 query efficiency?

### Phase 3 — CMA-ES Optimization
- [ ] Which CMA-ES variant minimizes queries on CIFAR-10? Compare: standard CMA-ES, (1+1)-CMA-ES, separable/diagonal CMA-ES, VD-CMA, LM-CMA.
- [ ] Which hyperparameters drive the most query variance: population size λ, initial step-size σ₀, recombination weights?
- [ ] What improvements are possible given the binary fitness signal? Candidates: surrogate fitness estimation, fitness shaping, restart strategies (IPOP/BIPOP).
- [ ] Does warm-starting CMA-ES with a structured covariance prior (e.g. from prior successful attacks on similar images) reduce queries?

### Decision Boundary & Search Space
- [ ] How complex are the decision boundaries of the CIFAR-10 classifier? Are they locally linear (favorable for binary search) or highly curved?
- [ ] How frequently does the predicted wrong label change during Phase 3 optimization (label switching)? Does instability correlate with attack difficulty or query count?
- [ ] Does the perturbation trajectory stay near the decision boundary throughout Phase 3, or does it drift into the interior of the adversarial region?

### Targeted Attacks
- [ ] Implement targeted attacks: measure the query overhead of forcing `f(x + δ) = y_t` vs untargeted, across all pairs of source/target classes on CIFAR-10.
- [ ] Does query cost depend on semantic distance between source and target class (e.g. cat → dog vs cat → airplane)?
- [ ] Can we exploit implicit structure in the hard-label feedback (e.g. sequence of label switches during optimization) to guide the search toward a specific target class more efficiently?

### Ablation & Baselines
- [ ] Run covariance ablation (full vs diagonal vs no adaptation) on CIFAR-10 — does the advantage of full CMA-ES shrink under very tight query budgets?
- [ ] Add a Genetic Algorithm (GA) baseline and compare query efficiency vs CMA-ES across attack success rate buckets.
- [ ] Measure the independent contribution of initialization strategy (random Gaussian vs low-frequency vs patch-based) to Phase 3 convergence, controlling for Phase 1/2 quality.

### Model variants & transfer analysis
- [ ] Test adversarial example transferability: do perturbations found against RobustBench WRN-28-10 transfer to the standard PyTorch model, and vice versa?
- [ ] Evaluate attack success rate and query cost difference between the two models — quantify the robustness advantage of adversarial training.
- [ ] Can insights from one model accelerate attacks on the other (e.g. warm-start initialization)?

### Robustness & Transferability
- [ ] Test whether adversarial examples found by EvolBA transfer to a different model architecture trained on the same CIFAR-10 data.
- [ ] Evaluate attack success rate against adversarially trained models — does query efficiency degrade gracefully or collapse?

