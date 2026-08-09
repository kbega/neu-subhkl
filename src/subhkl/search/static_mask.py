"""Static-structure masks: what the detector always sees is not a peak.

Beam-stop shadows, illumination boundaries and instrument-body glow are fixed
in the detector frame, while a Bragg reflection survives at most a frame or
two of sample rotation.  A per-bank low quantile across enough frames
(exposure-normalised) therefore contains the artifacts and none of the crystal -- the frames do not even have
to come from the same sample, only from the same instrument configuration,
which makes the mask an instrument calibration product rather than a
per-dataset one.

Two artifact classes, measured on CG4D l1-mbl (both recurring across ten
runs while every Bragg peak moved):

* an illumination boundary -- the static background fell 2.2 -> 0.6 counts
  over ~25 columns, a gradient 8x the panel's typical column-to-column
  variation.  The smooth background model cannot follow a step, so the
  residual ridge along it is tiled with false atoms.  The boundary sat
  ~190 px inside the panel, out of reach of any border erosion.
* a diffuse static glow at ~2x the ambient rate, carrying dozens of weak
  spurious detections whose positions jitter from run to run -- invisible
  to any peak-level recurrence veto, caught only by a pixel mask.

The mask marks the union of a gradient criterion (boundaries) and a level
criterion (glow) on the smoothed static map, dilated so that atoms merely
touching the structure are covered.

The file format is a reduced single-frame stack: ``images`` [n_banks, H, W]
(uint8, 1 = valid, 0 = masked) with ``bank_ids`` alongside, so every tool
that reads reduced files can display a mask, and the finder maps mask frames
onto its input by physical bank rather than by position in the file.
"""

from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np
from scipy import ndimage


def estimate_static_mask(
    frames: np.ndarray,
    *,
    smooth_sigma: float = 2.0,
    grad_nmads: float = 8.0,
    glow_factor: float = 2.0,
    dilate_px: int = 8,
    static_quantile: float = 25.0,
    grad_min_frac: float = 0.02,
) -> np.ndarray:
    """Valid-pixel mask (1 = usable) for one bank from its frame stack.

    Two robustness properties are load-bearing:

    * Frames are normalised to unit mean rate before stacking, so the same
      static feature at a different exposure (or beam current) produces the
      same mask, and frames of mixed exposures may be pooled.  Both criteria
      below are then ratios on a relative rate map rather than counts.
    * The static map is a *low quantile* across frames (default p25), not
      the median.  A dense diffraction pattern puts some reflection near a
      given pixel in half the frames, which pollutes a median built from a
      handful of runs -- measured on l1-mbl forward banks, where the median
      map's gradients masked genuine Bragg peaks.  A static feature is
      present in *every* frame, so it survives any quantile; a reflection
      would have to sit still through >75% of the scan to leak in.

    ``grad_nmads`` is the boundary criterion: pixels where the static map's
    gradient magnitude exceeds this many MADs of the panel-wide gradient are
    structure, not statistics (the l1-mbl illumination edge measures ~8 MADs
    of margin even at threshold 8).  ``glow_factor`` is the level criterion,
    relative to the panel's ambient rate.  ``dilate_px`` should cover an atom
    footprint (~2 * max_sigma) so that an atom whose tail rests on the
    structure is masked along with it.
    """
    frames = np.asarray(frames, dtype=np.float32)
    if frames.ndim != 3 or frames.shape[0] < 2:
        raise ValueError("need a [n_frames, H, W] stack with at least 2 frames")

    scales = frames.mean(axis=(1, 2), keepdims=True)
    rates = frames / np.maximum(scales, 1e-6)
    # Smooth each frame *before* the quantile: at counting rates near one
    # photon per pixel a low quantile of raw integers is zero on both sides
    # of any structure, and the structure vanishes with it.  Smoothing first
    # turns each frame into a local rate estimate, whose low quantile keeps
    # what every frame shows.
    smooth = np.percentile(
        np.stack([ndimage.gaussian_filter(f, smooth_sigma) for f in rates]),
        static_quantile,
        axis=0,
    )

    gy, gx = np.gradient(smooth)
    grad = np.hypot(gy, gx)
    ambient = max(float(np.median(smooth)), 1e-3)

    # Two conditions, one statistical and one physical.  MADs alone are a
    # pure significance test: on a genuinely flat forward-scattering panel
    # the gradient MAD is the noise floor, and soft real variation (diffuse
    # scattering, halo skirts) clears any number of MADs -- measured on
    # l1-mbl forward banks, where significance alone masked 12-17% of the
    # panel and with it genuine Bragg peaks.  The floor demands an actual
    # step: the l1-mbl illumination boundary runs ~3% of ambient per pixel,
    # halo gradients well under 1%.
    grad_mad = np.median(np.abs(grad - np.median(grad))) + 1e-9
    boundary = (np.abs(grad - np.median(grad)) > grad_nmads * grad_mad) & (
        grad > grad_min_frac * ambient
    )

    glow = smooth > glow_factor * ambient

    bad = boundary | glow
    if dilate_px > 0:
        bad = ndimage.binary_dilation(bad, iterations=int(dilate_px))
    return (~bad).astype(np.uint8)


def _read_frames_by_bank(paths: list[Path]) -> dict[int, list[np.ndarray]]:
    by_bank: dict[int, list[np.ndarray]] = {}
    for path in paths:
        with h5py.File(path, "r") as f:
            images = f["images"]
            bank_ids = np.asarray(f["bank_ids"][()]).astype(int)
            if len(bank_ids) != images.shape[0]:
                raise ValueError(
                    f"{path}: bank_ids ({len(bank_ids)}) does not match "
                    f"images ({images.shape[0]})"
                )
            for i, bank in enumerate(bank_ids):
                by_bank.setdefault(int(bank), []).append(
                    np.asarray(images[i], dtype=np.float32)
                )
    return by_bank


def build_mask_file(
    inputs: list[str | Path],
    output: str | Path,
    *,
    min_frames: int = 5,
    smooth_sigma: float = 2.0,
    grad_nmads: float = 8.0,
    glow_factor: float = 2.0,
    dilate_px: int = 8,
    static_quantile: float = 25.0,
    grad_min_frac: float = 0.02,
) -> dict:
    """Estimate one mask per physical bank across every input file.

    Inputs are reduced/merged stacks (``images`` + ``bank_ids``); they may
    come from different samples.  A bank with fewer than ``min_frames``
    frames gets a fully valid mask -- stated in the summary rather than
    silently guessed from statistics too thin to tell a peak from a shadow.
    """
    by_bank = _read_frames_by_bank([Path(p) for p in inputs])
    if not by_bank:
        raise ValueError("no frames found in the inputs")

    banks = sorted(by_bank)
    masks, thin = [], []
    for bank in banks:
        frames = np.stack(by_bank[bank])
        if frames.shape[0] < min_frames:
            masks.append(np.ones(frames.shape[1:], dtype=np.uint8))
            thin.append(bank)
        else:
            masks.append(
                estimate_static_mask(
                    frames,
                    smooth_sigma=smooth_sigma,
                    grad_nmads=grad_nmads,
                    glow_factor=glow_factor,
                    dilate_px=dilate_px,
                    static_quantile=static_quantile,
                    grad_min_frac=grad_min_frac,
                )
            )

    stack = np.stack(masks)
    with h5py.File(output, "w") as f:
        f["images"] = stack
        f["bank_ids"] = np.asarray(banks, dtype=np.int64)
        f.attrs["kind"] = "static-mask"
        f.attrs["inputs"] = [str(p) for p in inputs]
        f.attrs["min_frames"] = min_frames
        f.attrs["smooth_sigma"] = smooth_sigma
        f.attrs["grad_nmads"] = grad_nmads
        f.attrs["glow_factor"] = glow_factor
        f.attrs["dilate_px"] = dilate_px
        f.attrs["static_quantile"] = static_quantile
        f.attrs["grad_min_frac"] = grad_min_frac

    masked_frac = 1.0 - stack.mean()
    return {
        "banks": banks,
        "n_frames": {b: len(by_bank[b]) for b in banks},
        "thin_banks": thin,
        "masked_fraction": float(masked_frac),
        "output": str(output),
    }


def load_mask_for_banks(
    path: str | Path, bank_ids: list[int], shape: tuple[int, int]
) -> np.ndarray:
    """Per-image validity stack for ``bank_ids``, mapped by physical bank.

    A bank absent from the mask file is fully valid: masks may be built from
    a subset of the instrument, and refusing to run would make the mask a
    gate rather than a filter.  A shape mismatch is an error -- it means the
    mask was built for a different reduction geometry, and applying it
    anyway would mask the wrong pixels.
    """
    with h5py.File(path, "r") as f:
        masks = np.asarray(f["images"][()], dtype=np.float32)
        file_banks = np.asarray(f["bank_ids"][()]).astype(int)
    if masks.shape[1:] != tuple(shape):
        raise ValueError(
            f"mask frames are {masks.shape[1:]}, input images are {tuple(shape)}; "
            "the mask was built for a different reduction geometry"
        )
    lookup = {int(b): masks[i] for i, b in enumerate(file_banks)}
    ones = np.ones(shape, dtype=np.float32)
    return np.stack([lookup.get(int(b), ones) for b in bank_ids])
