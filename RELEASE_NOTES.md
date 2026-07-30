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

### Changed

- **Default `gamma` is now 0.5, was 1.0** (2.0 in the orchestrator). `gamma=1`
  is the value at which the penalty per unit flux becomes independent of scale,
  so one broad atom and a mass-preserving spread of narrower ones have the same
  cost and the same predicted image; the minimiser is then not unique in the
  scale coordinate and the fit breaks the tie towards splitting. See
  `docs/matrix_free_theory.md` Theorem 1. The legacy finder keeps its historical
  default of 2.0 so that it still reproduces what it always did.
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
- The outer convergence test measured the raw Newton direction rather than the
  step actually taken, so it never triggered.
- `subhkl finder --finder-algorithm sparse_rbf` returned no peaks on its own
  integration test. It now passes.
- Removed `src/subhkl/:q`, a stray editor buffer saved under the wrong name.

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
- Two heavily overlapping broad peaks — 2.67 sigma of separation in
  `test_overlapping_ghost_center_shift_failure`, marked `xfail` — are not
  reconstructed reliably: repeated runs return either the correct two atoms or
  those plus a spurious one at the composite centre, with recovered flux from
  0.77x to 2.6x truth. This is *not* a conditioning limit; the Gram matrix of
  the true pair has a condition number of 1.40 and that of the realised support
  1.5. The reported model simply does not fit the data there, with an rms
  residual of 23.8 against 7.7 for the background alone.
- The global `Deviance/DoF` statistic does not detect the case above (1.21,
  against 1.12 for a well-behaved control) because it is diluted across the
  whole image. Evaluated over a peak's own footprint the same statistic gives
  26.9 against 1.04, so a per-peak goodness-of-fit check would catch it. Not yet
  wired to anything.
- The morphological background estimator under-fits smooth extended structure at
  its centre — by about 21% on a diffuse halo — leaving a broad positive
  residual. The boundary-sigma rejection above removes the resulting artefacts
  from the peak list, but does not fix the estimate itself.
