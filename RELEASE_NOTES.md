# Release Notes

All notable changes to this project are recorded here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project
adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- Global basis-pursuit peak finder (`MatrixFreeSparseRBFPeakFinder`) is now
  reachable from the command line and is the default for
  `--finder-algorithm sparse_rbf`.
- Continuous ("sliding") refinement of the selected support: after the convex
  solve chooses which atoms are present, their amplitude, position and width are
  re-fitted against the same Poisson objective, so recovered positions are
  sub-pixel rather than sub-grid. Reaches 0.03–0.17 px on an isolated synthetic
  peak. Controlled by `refine_positions`, default on.
- Rejection of atoms whose fitted width reaches the edge of the sigma bank. Such
  an atom is the solver asking for a wider basis than it was given, which is
  what unmodelled smooth background looks like rather than a reflection.
  Controlled by `reject_boundary_sigma` and `boundary_sigma_frac`, default on.
- The count of atoms rejected that way is recorded on `n_boundary_rejected` and
  reported under `show_steps`, rather than being discarded silently.
- Multiplicity correction to the significance threshold. Solving globally tests
  every (pixel, scale) coefficient at once, so `alpha` is now floored at
  `sqrt(2 log N_k)` with `N_k` the resolution-element count for that scale.
- `docs/matrix_free_theory.md`, recording the results behind these changes with
  proofs and the measurements that establish them.
- `--sparse-rbf-legacy`, to opt back out to the greedy finder.
- `alpha` may now be left as `None`, which is the new default. The significance
  threshold is then derived from the data: the level at which the expected
  number of false detections over the whole image is O(1), from the maximum of a
  smooth Gaussian field over the resolution elements at each scale. Passing a
  number keeps it as a lower bound on significance, so an explicit request can be
  stricter than false-alarm control requires but not weaker. Exposed as
  `effective_alpha(height, width)`.
- `debias`, to control the post-selection refit. Off by default: see below.

### Changed

- **Default `gamma` is now 0.5, was 1.0** (2.0 in the orchestrator). `gamma=1`
  is the value at which the penalty per unit flux becomes independent of scale,
  so one broad atom and a mass-preserving spread of narrower ones have the same
  cost and the same predicted image; the minimiser is then not unique in the
  scale coordinate and the fit breaks the tie towards splitting. See
  `docs/matrix_free_theory.md` Theorem 1. The legacy finder keeps its historical
  default of 2.0 so that it still reproduces what it always did.
- **Debiasing is now off by default.** It exists to remove L1 shrinkage from
  amplitudes, and nothing downstream reads the finder's amplitude — the
  orchestrator reduces the peak list to `(row, column)` before handing it to the
  workers, and intensity is measured later by the integrator at known positions,
  where it is a well-posed problem. It is also not free: dropping the penalty
  drops the only thing suppressing what the model cannot explain, so an
  unpenalised refit absorbs a mis-estimated background into the peaks. Turning it
  off removes two flaky tests and takes a clean full-suite run from roughly 55%
  of runs to roughly 87%. Set `debias=True` when amplitudes are wanted from the
  finder itself.
- **Default `alpha` is now `None`, was 4.0.** The right threshold depends on how
  many coefficients are being tested, so it depends on image size, which a
  constant cannot express. The derived values run from 3.60 on a 64x64 padded
  crop to 5.44 on a 4096x4096 frame, so the old constant was about right for a
  mid-size image but ~26% below the floor on a full detector — under-thresholded
  exactly where it matters. The `sigma**gamma` shape of the threshold is kept,
  since that is what sets the merge/split balance; deriving the level from the
  floor *alone* flattens that shape into the over-merging regime and loses weak
  peaks in the tails of strong ones, which was measured and rejected.
- Peak positions are now reported from the coefficient-weighted centroid of the
  raw coefficients rather than a log-parabola on the smoothed map, which was
  dragging centres toward neighbouring peaks.
- Detection smoothing in peak extraction is capped at the finest scale in the
  bank, so it no longer merges peaks a few pixels apart.
- Peaks are ranked and truncated to the reporting limit *after* out-of-bounds
  ones are discarded, not before, so maxima in the replicated edge padding can
  no longer consume the budget and displace real interior peaks.

### Fixed

- Conjugate gradients was being given a non-symmetric operator in both the
  semi-smooth Newton solve and the debiasing solve, because the Jacobi scaling
  was multiplied into the operator instead of passed as a preconditioner.
  Measured asymmetry of the operators as written was ~50%. The debiasing solve
  diverged to NaN as a result, and because a non-finite iterate was allowed to
  propagate, whole images returned no peaks at all.
- The L1 threshold used the across-channel maximum of the Hessian diagonal as
  the coefficient variance. The noise on a channel's coefficient is set by that
  channel's own curvature; using the maximum understated it by ~11x for the
  narrowest basis in a typical bank, so fine scales were thresholded far too
  weakly and fitted noise.
- The prox-gradient step size was derived from the diagonal of `A^T W A` rather
  than a bound on its largest eigenvalue. For a typical bank the two differ by a
  factor of 419, so every step overshot and the line search collapsed onto its
  smallest permitted step, leaving the solve stalled far from the optimum.
- The line search accepted a step that increased the objective when its
  backtracking budget ran out, instead of rejecting it.
- The debiasing phase diverged on large, heavily overlapping supports. It drops
  the L1 term, which is the only thing holding the near-null-space directions of
  the active set in check, so on a near-singular support CG returned a direction
  it had not solved for and the unguarded Newton step ran away. Measured on two
  overlapping broad peaks, the likelihood got *worse* on every iteration and the
  rms residual went from 3.3 to 60 — an order of magnitude worse than reporting
  no peaks at all, while the L1 phase that preceded it had fitted the data well.
  Debiasing now backtracks and refuses a step that does not improve the
  likelihood, so it can only ever improve on the L1 solution it starts from.
- The outer convergence test measured the raw Newton direction rather than the
  step actually taken, so it never triggered.
- `subhkl finder --finder-algorithm sparse_rbf` returned no peaks on its own
  integration test. It now passes.
- Two heavily overlapping broad peaks at 2.67 sigma of separation were
  reconstructed with a spurious atom at the composite centre and up to 2.6x the
  true flux (`test_overlapping_ghost_center_shift_failure`). This was the
  debiasing divergence above, not a resolution limit: the condition number of
  the true pair is 1.40. The test now passes.
- Removed `src/subhkl/:q`, a stray editor buffer saved under the wrong name, and
  added editor swap files to `.gitignore` so that cannot recur.

### Notes

- Debiasing is load-bearing for amplitude accuracy only, which is why it is no
  longer the default. The one test that needs it,
  `test_poisson_vs_gaussian_sparse_flux`, now opts in explicitly: it compares
  recovered flux against ground truth on a very low-count image (background rate
  0.1, peak amplitude 5.0, true flux 125.7), and without the refit only 6.4 of
  that flux survives L1 shrinkage — 5% of the truth — which swamps the
  difference between the two losses it exists to measure.
- Measured flake rates, whole file, per test: with debiasing on,
  `test_poisson_local_variance_suppression` passed 5 runs of 8 and
  `test_poisson_subpatch_variance_suppression` 7 of 8; with it off both are
  stable and `test_peak_finder_multiscale_subpixel_recovery` passes 7 of 8
  instead. Amplitude quality feeds back into position through the sliding
  refinement's starting point, so the residual flake is not surprising.

### Deprecated

- `SparseRBFPeakFinder`, the greedy matching-pursuit finder, is superseded by
  the basis-pursuit finder and reachable only via `--sparse-rbf-legacy`. It
  cannot be removed yet: `SparseLaueIntegrator` still inherits from it to obtain
  the peak model. Breaking that inheritance, by extracting the shared model into
  its own module, is the prerequisite.

### Known issues

- Peak finding is not reproducible run to run on GPU. Identical input and flags
  return one of a small number of distinct peak sets, because reductions are
  not deterministic; `XLA_FLAGS=--xla_gpu_deterministic_ops=true` makes it
  bit-reproducible at about 2.1x wall clock. Tracked in #13. Tests whose
  assertions sit near a threshold will flap accordingly.
- `test_poisson_local_variance_suppression` and
  `test_poisson_subpatch_variance_suppression` each pass in isolation (six runs
  out of six) but fail roughly one run in three when the whole file is run, so
  their outcome depends on what executed before them. That is the
  non-determinism above expressing itself through JIT and GPU state rather than
  through the test inputs.
- The global `Deviance/DoF` statistic is too diluted to detect a locally bad
  fit: on a case where the reported model was worse than reporting no peaks at
  all it read 1.21, against 1.12 for a well-behaved control. Evaluated over a
  peak's own footprint the same statistic gave 26.9 against 1.04. A per-peak
  goodness-of-fit check would therefore catch what the global one misses, and is
  not yet wired to anything.
- The morphological background estimator under-fits smooth extended structure at
  its centre — by about 21% on a diffuse halo — leaving a broad positive
  residual. The boundary-sigma rejection above removes the resulting artefacts
  from the peak list, but does not fix the estimate itself.
