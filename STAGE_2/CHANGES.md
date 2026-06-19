# Stage 2 — Design decisions log

(Mirrors `STAGE_1/CHANGES.md`'s role: records *why* a direction was taken or
discarded, not just what the code does.)

## Full CMA-ES discarded from the ES-variant comparison

Stage 2's first study compares sep-CMA-ES (current baseline), VkD-CMA
(rank 1/2/3), and (1+1)-CMA-ES. "Full CMA-ES" (a dense n x n covariance,
n=3072 for CIFAR-10) was considered and discarded for this round.

**Why it's not just "more generations" — it's the same generations, each
doing far more work.**

The query budget and `lam` (offspring per generation) are unchanged by the
choice of covariance representation, so the *number of generations*
(~16-34, set by `max_queries / (lam + overhead)`) is the same regardless of
whether the covariance is diagonal or full. What differs is the amount of
**non-query CPU work per generation**:

| operation | sep-CMA-ES (diagonal `D`) | full CMA-ES (`C` is 3072x3072) |
|---|---|---|
| covariance update / generation | O(n) ~ 3K ops | rank-mu update: O(mu*n^2) ~ 28 x 9.4M ~ 2.6e8 ops |
| sampling (need `C^{1/2}`) | O(n) (scale by `D`) | eigendecomposition: O(n^3) ~ 2.9e10 ops |

That's roughly 5-7 orders of magnitude more arithmetic per generation,
producing *zero* additional oracle queries. Across the 200-image x 2-model
protocol (400 runs), this plausibly adds hours of pure linear-algebra time
to a study that otherwise takes ~1h.

**Statistical argument (the original sep-CMA-ES rationale, BRAINSTORMING.md
section 1.1):** even if the compute were free, a rank-mu (mu~14-28) update
is estimating a 9.4M-entry matrix from ~14-28 samples per generation — almost
entirely noise. Diagonal (n params) and VkD (n + kn params, k<=3) are the
regimes where `lam~14-28` samples/generation can plausibly produce a
meaningful estimate.

**Decision:** discard full CMA-ES for this study. If VkD's results suggest
unexplained structure that a few extra rank-k directions can't capture, it
could be revisited later as a small-scale diagnostic (e.g. N=20 images, one
model) rather than the full protocol.

## EC1 result: sep-CMA-ES (k=0) wins -- VkD and (1+1) don't replace it

Ran EC1 (N=100, Q=2000, 1000 runs: sep/vd1/vd2/vd3/(1+1) x standard/robust).
Full results/analysis: `ec1_results.ipynb` (`outputs/ec1_full/`).

- **sep (k=0) has the best median best_l2 of all 5 conditions on both
  models.** vd1/vd2/vd3 are each slightly worse (0.0-0.3% median, 40-44%
  win rate vs sep, paired per-image) on standard, and noise-level neutral
  on robust. (1+1) is clearly worse on standard (-6.6% median, 36% win
  rate) and mixed on robust (58.9% win rate but worse median, due to a
  skewed distribution).
- VkD's V matrices converge to a **consistent ~1.05/0.80/0.64 norm pattern
  for columns 0/1/2, regardless of k or model** -- looks like a property of
  the self-normalised update dynamics (driven by `mu`/`mueff` eigenvalue-
  ratio statistics) rather than evidence of image-specific structure being
  captured.
- The mock (N=10, Q=300) showed the *opposite* trend (vd1->vd2->vd3 ticking
  slightly below sep) -- not representative; the short horizon (4-5 gens)
  doesn't reflect the full-budget regime (~30 gens).

**Decision:** keep `evolba_vkd(vk_rank=0)` (== `evolba_tuned`) as the
production configuration. Neither VkD nor (1+1)-CMA-ES are adopted.
Caveat: the carrier config was tuned *for* sep-CMA-ES in Stage 1 -- this
answers "does VkD/(1+1) help as a drop-in replacement under sep's tuned
settings", not "what is VkD/(1+1)'s ceiling under its own best settings".
Revisiting with per-variant tuning is a possible future direction but not
currently planned.
