# Changes to evolba_tuned.py / evolba_baseline.py (Stage 1)

## best_l2 now tracks the running minimum, not the final point

`evolba_tuned()` previously returned `best_l2 = ||m_final - x_orig||` -- the
L2 of the LAST generation's point. But CMA-ES here is non-elitist: `m = m_new`
happens unconditionally at the end of every generation, even if `m_new` is
farther from `x_orig` than `m` was (the `tau`-backtrack loop can fail to fully
recover a regression, and the old code accepted whatever came out anyway). So
the reported `best_l2` could reflect a regressed final point, not the best
boundary point actually visited.

Fixed:
  - `best_l2  = min(init_l2, min(l2 for _, l2 in trajectory))`  -- true best
  - `final_l2 = ||m_final - x_orig||`                            -- old value, kept for diagnostics

The search itself (how `m`, `D`, `xi` evolve) is UNCHANGED -- this only
affects what's reported as the attack's result.

Rationale: reporting best-ever is free and doesn't constrain the search,
whereas a temporarily-worse `m` can be a normal/healthy part of non-elitist
CMA-ES exploration (it still informs the covariance/step-size adaptation via
the offspring statistics). Conflating "current search state" with "best
result found" was simply a bug, not a tradeoff.

## tau (backtrack budget) grid search -- recommend tau=0 + best_l2 bookkeeping

Study 6 (`run_study6.py`, full: 1600 runs = 4 tau values x 200 images x 2
models x 2000 queries, see `study6_results.ipynb`) tested whether
backtracking (`tau>=1`) is worth its query cost, given that the `best_l2`
fix above makes "going in the wrong direction temporarily" free as long as
the search recovers later.

Result: `tau=0` (no backtracking) gives the best `best_l2` AND the most
completed generations for BOTH models (standard: IR 0.44 vs 0.35-0.37 for
tau>=1, 57 vs 44-50.5 gens; robust: IR 0.071 vs 0.048-0.052, 59 vs 12-18
gens). ~94-100% of regressions at tau=0 self-heal within ~46-72 extra
queries with zero dedicated recovery cost. For the robust model, tau>=1
triggers a feedback loop: backtrack queries -> fewer generations -> sample
skewed toward early high-step-size generations -> more regressions proposed
-> more backtracking -> even fewer generations (median gens collapse
59->18->12->12).

CAVEAT (robust model only): the tau=0 win depends entirely on best_l2
bookkeeping. Looking at the *current* (not best-ever) L2 at Q=2000, tau=0
(7.095) is actually WORSE than tau=2/3 (7.044/7.023) -- tau=0 produces a
high-variance trajectory that dips to a great minimum mid-run but wanders
back up by the end, while tau>=2 stays close to a smoother, worse minimum
throughout. If a deployment cannot track best-so-far and must return its
*current* adversarial example (e.g. a true anytime attack), tau=0 is the
worst choice for the robust model, at every checkpoint including the final
one. See the "Addendum" section of `study6_results.ipynb` for the full
table and discussion.

Recommendation: `tau=0` + best_l2 bookkeeping (already the default return
value) as the standard config; `tau>=2` only if a use case specifically
needs a low-variance "current point" anytime guarantee against robust
models.

## Study 7 -- individually-validated tuning choices do not compound

Studies 3/5/6 each validated one change in isolation, all under
`lam_override=14`: `xi_step_scale=0.5`, `tau=0` (+ best_l2 bookkeeping),
`bs_adaptive=True`. Study 7 (`run_study7.py`, full: 800 runs = 2 conditions
x 200 images x 2 models x 2000 queries, see `study7_results.ipynb`) combined
all three into a single "tuned" config and compared it against a faithful
reproduction of `evolba_baseline()`'s hyperparameters (`xi_step_scale=1.0,
tau=3, bs_steps=26` fixed, non-adaptive), with `lam_override=None` (=28,
baseline's default) held fixed in BOTH arms -- `lam` tuning was deliberately
deferred.

Result: the gains do **not** compound. `tuned` gets roughly double the
generations of `baseline` in both models (33-34 vs 16-20.5), but ends up a
wash for standard (IR 0.380 vs 0.386, 50% win rate) and *worse* for robust
(IR 0.047 vs 0.060, only 26.7% win rate, -0.9% median relative).

Why: comparing to Study 6 (same `xi_step_scale=0.5, tau=0`, but at
`lam_override=14`), that combo got 57-59 generations and IR 0.441/0.071.
Going to `lam=28` roughly halves the generation count as expected (lam sets
offspring-per-generation), but the IR *also* drops disproportionately
(0.441->0.380, 0.071->0.047) -- i.e. `xi_step_scale=0.5`'s "many small
steps" strategy was tuned for the generation-abundant `lam=14` regime.
`baseline`'s bigger steps (`xi_step_scale=1.0`) extract more progress per
generation, enough to match or beat `tuned` even with half the generations.

Implication: `lam` is not an independent knob that can be tuned in
isolation after the fact -- it determines which step-size regime
(`xi_step_scale`) is optimal. The Study 3/5/6 recommendations are valid *at
lam=14* but do not transfer cleanly to `lam=28`. The natural next study is a
joint `lam x xi_step_scale` sweep.

## Study 8 -- the optimal xi_step_scale depends on lam (joint sweep)

Study 7 found that `xi_step_scale=0.5` (Study 3's pick, validated at
`lam_override=14`) stops being a clear win at `lam_override=None` (=28).
Study 8 (`run_study8.py`, full: 1800 runs = 9 conditions x 100 images x 2
models x 2000 queries, see `study8_results.ipynb`) sweeps both knobs
jointly: `LAM_VALUES=[14,28,42] x XI_VALUES=[0.5,0.75,1.0]`, carrier held
fixed at `tau=3, bs_adaptive=True, bs_cap=26, cmu_scale=1.0` (same as Study
7's `tuned`/EC1's `sep`). `L28_X050` is exactly that carrier config -- a
useful reference point.

**Caveat**: this grid uses `tau=3`, not Study 6's `tau=0`. Study 6's best
validated config (`lam=14, xi_step_scale=0.5, tau=0` -> IR=0.441/0.071) is
not reproduced here -- `L14_X050` at `tau=3` only gets IR=0.338/0.050. This
study is about the `(lam, xi)` interaction at fixed `tau=3`, not the overall
best across all four knobs.

Result: **the optimal `xi_step_scale` shifts with `lam`.**
- Standard model: `xi=0.5` is best only at `lam=14` (IR=0.3381). At
  `lam=28` and `lam=42`, `xi=0.75` wins instead, and `L42_X075`
  (IR=0.3601) is the best single cell in the whole grid.
- Robust model: `lam=14, xi=0.75` (IR=0.0706) is a standout -- the only
  cell that approaches Study 6's `tau=0` robust IR (0.071) despite `tau=3`
  here. `lam=28` is uniformly the worst row (0.044-0.055) regardless of
  `xi`.
- Paired vs `L28_X050`: 8 of 9 conditions are >= as good on both models
  (win rates 50-70%, median relative improvement up to +5.5%); only
  `L42_X050` is worse (44.0%/48.9% win rate). `L28_X050` (the Study
  7/EC1 carrier) is close to the worst point in this grid.

No single `(lam, xi)` pair dominates both models: `L14_X075` is best for
robust but mid-table for standard; `L42_X075` is best for standard but only
middling for robust. `L14_X050` remains the best balance across both,
though still below Study 6's `tau=0` combination.

Implication: `lam` and `xi_step_scale` interact, but `tau` is a third axis
not explored in this grid. Natural next step: revisit `tau in {0,3}` at the
`(lam, xi)` combinations that did best here (`L14_X075`, `L42_X075`),
deferred to a future study.

## binary_search_adaptive (float32-saturation early stop)

Added `binary_search_adaptive()` in evolba_baseline.py: stops the bisection
early once `mid` rounds (in float32) to exactly `lo` or `hi` -- lossless
(returns the same `hi` as fixed-step `binary_search`), see
`outputs/diag_bs_adaptive/findings.md` for validation. Wired into
`evolba_tuned` via `bs_adaptive`/`bs_cap`/`do_binary_search`.
