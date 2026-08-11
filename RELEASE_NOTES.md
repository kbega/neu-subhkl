# Release Notes

All notable changes to this project are recorded here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project
adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.3.0]

The peak finder is rebuilt around a global convex solve: it fits all peaks at
once instead of subtracting them one at a time. Most of that is invisible, but
the defaults changed and one output column changed meaning, so scripts that pin
finder settings or read its HDF5 need a look.

- **`--finder-algorithm sparse_rbf` now runs the new basis-pursuit finder**,
  with sub-pixel position refinement. `--sparse-rbf-legacy` restores the old
  greedy one.
- **The basis now sizes itself**: `--sparse-rbf-max-sigma` and
  `--sparse-rbf-num-sigmas` default to unset and are measured from your first
  batch. A hand-set ceiling that is too small splits one peak into several, so
  prefer the default unless you know the width.
- **Sensitivity is now one number**: `--sparse-rbf-false-alarms-per-image`
  (default 0.1), the expected count of spurious peaks per image — lower it to
  demand more evidence. Related defaults changed: `--sparse-rbf-gamma`
  1.0 → 0.0, and `--sparse-rbf-loss` gaussian → poisson, since detector frames
  are photon counts.
- **`peaks/sigma` in the finder's output is now the fitted peak width in
  pixels, not the intensity sigma** — same name, different quantity, so check
  anything reading it. Two per-peak fit statistics join it: `peaks/deviance`
  (is this peak real?) and `peaks/residual_deviance` (is it fitted well?).
  `--no-hull-filter` reports every peak the finder proposes, for diagnosis.
- **Plots can be redrawn after the fact**: `finder-visualize` and
  `integrator-visualize` rebuild the unrolled-detector plots from existing
  HDF5, so a run can skip plotting and still be inspected later, at any `--dpi`.
- **Detector artifacts can be masked**: `static-mask` builds a mask of static
  structure (dead panels, glow, shadows) from your frames, applied with
  `finder --static-mask-file`; `sum-images` and `mask-visualize` support it.
- **`--multi-gpu` shards the finder and indexer across all visible GPUs.**
  Opt-in, because JAX claims memory on every visible device.

### Known issues

- Peak finding is not bit-reproducible on GPU: identical inputs can return
  slightly different peak sets. `XLA_FLAGS=--xla_gpu_deterministic_ops=true`
  makes it reproducible at about 2.1x wall clock (#13).
