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

The mask marks static structure on a *band-pass* of the static map --
scales between the atom footprint and the background window -- because that
is precisely the structure the background model cannot follow and the finder
mistakes for peaks.  A wide smooth halo passes: the background follows it,
and the genuine reflections sitting on it must stay findable.  Steps are
caught by the band-pass gradient, textured glow by the band-pass level, and
the union is dilated so atoms merely touching structure are covered.

The file format is a reduced single-frame stack: ``images`` [n_banks, H, W]
(uint8, 1 = valid, 0 = masked) with ``bank_ids`` alongside, so every tool
that reads reduced files can display a mask, and the finder maps mask frames
onto its input by physical bank rather than by position in the file.
"""

from __future__ import annotations

import hashlib
from collections import Counter
from pathlib import Path

import h5py
import numpy as np
from scipy import ndimage


def estimate_static_mask(
    frames: np.ndarray,
    *,
    smooth_sigma: float = 2.0,
    grad_nmads: float = 8.0,
    texture_factor: float = 0.25,
    wide_sigma: float = 10.0,
    dilate_px: int = 8,
    static_quantile: float = 25.0,
    grad_min_frac: float = 0.02,
    clear_disks: list | None = None,
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
    of margin even at threshold 8).  ``texture_factor`` is the band-pass level
    criterion, relative to the panel's ambient rate; ``wide_sigma`` sets the
    long end of the band and should sit at the background window's scale
    (the finder's is max(15, 5 * max_sigma) px wide).  ``dilate_px`` should cover an atom
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
    # method="lower": a true order statistic.  The default interpolated
    # percentile blends adjacent order statistics, so a bright reflection
    # present in only some frames leaks a *fraction* of itself into the
    # static map -- and a 400-count peak leaking ten percent still dwarfs
    # any threshold stated in units of a ~2-count ambient.  Measured on
    # l1-mbl forward banks as an exclusion ring around every true peak.
    stack = np.stack([ndimage.gaussian_filter(f, smooth_sigma) for f in rates])

    # Exonerated peaks leave the evidence pool: ``clear_disks`` carries, per
    # frame, the footprints of detections whose fit metrics certify them as
    # genuine (see build_mask_file).  Removing the footprint from *this
    # frame's* evidence -- rather than subtracting a model, which would
    # leave mismatch dipoles -- means a real reflection cannot be declared
    # static however many frames it persists through, which is exactly what
    # happens when a manually oriented crystal repeats its Laue zones.
    if clear_disks is not None:
        rr, cc = np.mgrid[0 : stack.shape[1], 0 : stack.shape[2]]
        for fi, disks in enumerate(clear_disks):
            for r0, c0, rad in disks:
                stack[fi][(rr - r0) ** 2 + (cc - c0) ** 2 <= rad**2] = np.nan

    import warnings as _warnings

    with _warnings.catch_warnings():
        _warnings.simplefilter("ignore", category=RuntimeWarning)
        smooth = np.nanpercentile(stack, static_quantile, axis=0, method="lower")
    # A pixel with too little surviving evidence proves nothing static.
    n_eff = np.sum(~np.isnan(stack), axis=0)
    smooth = np.where((n_eff >= 2) & np.isfinite(smooth), smooth, 0.0)

    ambient = max(float(np.median(smooth)), 1e-3)

    # The criterion lives on a *band-pass* of the static map: structure at
    # scales between the atom footprint (smooth_sigma) and the background
    # window (wide_sigma).  What the mask must mark is structure that
    # defeats the background model -- and only that.  A wide static halo,
    # however elevated or statically steep its shoulders, is followed by
    # the windowed background estimate and is where real Bragg peaks live;
    # masking it (as an absolute-gradient or absolute-level criterion does)
    # removed genuine reflections from the centre of the l1-mbl forward
    # banks.  The halo vanishes from the band-pass; what remains is exactly
    # the peak-confusable statics: the illumination step (band-pass swing
    # ~0.4x ambient), the plume's texture (~0.3x at its false-detection
    # sites, against ~0.13x under real forward-bank peaks).
    band = smooth - ndimage.gaussian_filter(smooth, wide_sigma)
    # Positive lobes only.  False peaks are *positive* unmodelled structure;
    # |band| would additionally mask the negative moat that the wide
    # subtraction digs around anything bright -- a ~wide_sigma-to-2*wide_sigma
    # ring, which on the forward banks turned every leak into a ~20 px
    # exclusion zone around a genuine reflection.
    texture = band > texture_factor * ambient

    # Steps also announce themselves as band-pass gradients; MADs alone are
    # a pure significance test whose floor on a flat panel is noise, so the
    # physical floor (gradient per pixel as a fraction of ambient) guards
    # it: the l1-mbl illumination boundary runs ~3% of ambient per pixel.
    gy, gx = np.gradient(band)
    grad = np.hypot(gy, gx)
    grad_mad = np.median(np.abs(grad - np.median(grad))) + 1e-9
    # Restricted to the positive side of the band: false atoms form on
    # positive residual structure.  The negative moat that the wide
    # subtraction digs around anything bright has gradients too, and
    # without this restriction they masked a ring around every static
    # spot.
    boundary = (
        (np.abs(grad - np.median(grad)) > grad_nmads * grad_mad)
        & (grad > grad_min_frac * ambient)
        & (band > 0.0)
    )

    bad = boundary | texture
    if dilate_px > 0:
        bad = ndimage.binary_dilation(bad, iterations=int(dilate_px))
    return (~bad).astype(np.uint8)


def _read_frames_by_bank(
    paths: list[Path],
) -> tuple[dict[int, list[np.ndarray]], dict[int, int]]:
    """Frames grouped by physical bank, one per distinct orientation.

    The estimator's entire premise is that the sample moves between frames.
    Feed it the same goniometer orientation twice -- the same file listed
    again, or a re-exposure at identical angles -- and the true signal is
    static by construction, so it would be promoted into the mask.  Frames
    are therefore deduplicated per bank: by goniometer angles when the file
    carries them (which also catches re-exposures with fresh counting
    noise), by content hash otherwise (which catches literal duplicates).
    The dropped counts are returned so the caller can say so.
    """
    by_bank: dict[int, list[np.ndarray]] = {}
    provenance: dict[int, list[tuple[int, int]]] = {}
    dropped: Counter = Counter()
    hashes: set = set()
    signatures: dict[int, list[tuple]] = {}
    for file_index, path in enumerate(paths):
        with h5py.File(path, "r") as f:
            images = f["images"]
            bank_ids = np.asarray(f["bank_ids"][()]).astype(int)
            if len(bank_ids) != images.shape[0]:
                raise ValueError(
                    f"{path}: bank_ids ({len(bank_ids)}) does not match "
                    f"images ({images.shape[0]})"
                )
            angles = None
            if (
                "goniometer/angles" in f
                and f["goniometer/angles"].shape[0] == images.shape[0]
            ):
                angles = np.asarray(f["goniometer/angles"][()], dtype=float)
            for i, bank in enumerate(bank_ids):
                frame = np.asarray(images[i], dtype=np.float32)
                digest = (int(bank), hashlib.sha1(frame.tobytes()).hexdigest())
                if digest in hashes:
                    dropped[int(bank)] += 1
                    continue
                ang = None if angles is None else tuple(np.round(angles[i], 4))
                sig = _scene_signature(frame)
                bank_sigs = signatures.setdefault(int(bank), [])
                if any(
                    (ang is None or prev_ang is None or ang == prev_ang)
                    and _same_scene(sig, prev_sig)
                    for prev_ang, prev_sig in bank_sigs
                ):
                    dropped[int(bank)] += 1
                    continue
                hashes.add(digest)
                bank_sigs.append((ang, sig))
                by_bank.setdefault(int(bank), []).append(frame)
                provenance.setdefault(int(bank), []).append((file_index, i))
    return by_bank, dict(dropped), provenance


def _scene_signature(frame: np.ndarray, block: int = 8) -> tuple:
    """Exposure-normalised block means, with the per-block noise scale."""
    H, W = frame.shape
    h, w = H // block, W // block
    scale = max(float(frame.mean()), 1e-6)
    rate = frame[: h * block, : w * block] / scale
    blocks = rate.reshape(h, block, w, block).mean(axis=(1, 3))
    # Poisson sd of a block mean of the *normalised* rate.
    noise = np.sqrt(np.maximum(blocks, 1e-3) / (block * block * scale))
    return blocks, noise


def _same_scene(sig_a: tuple, sig_b: tuple) -> bool:
    """Whether two frames are consistent with re-exposures of one scene.

    Strict on purpose: the cost of a false "duplicate" is silently thrown
    away data, so only agreement everywhere within counting noise counts.
    The max of ~4k standardised block differences between true re-exposures
    sits near 4 sigma; a moved crystal shifts bright peaks by many blocks
    and lights up tens of sigma.
    """
    a, na = sig_a
    b, nb = sig_b
    if a.shape != b.shape:
        return False
    z = np.abs(a - b) / np.sqrt(na**2 + nb**2 + 1e-12)
    return float(z.max()) < 6.0


def _confident_peaks_by_frame(
    peaks_path: str | Path,
    deviance_min: float,
    residual_max: float,
    clear_nsigmas: float,
    margin: float,
) -> dict[int, list[tuple[float, float, float]]]:
    """Per image index: (row, col, clear radius) of metric-certified peaks.

    The certificate is the finder's own fit statistics: enough evidence
    (per-peak deviance well above the chi^2_4 admission level) and a shape
    the atom family explains (residual deviance per DoF near one).  An
    artifact fails one or both and earns no exoneration.
    """
    out: dict[int, list[tuple[float, float, float]]] = {}
    with h5py.File(peaks_path, "r") as f:
        idx = np.asarray(f["peaks/image_index"][()]).astype(int)
        r = np.asarray(f["peaks/pixel_r"][()], dtype=float)
        c = np.asarray(f["peaks/pixel_c"][()], dtype=float)
        sigma = np.asarray(f["peaks/sigma"][()], dtype=float)
        dev = np.asarray(f["peaks/deviance"][()], dtype=float)
        res = np.asarray(f["peaks/residual_deviance"][()], dtype=float)
    confident = (dev > deviance_min) & (res < residual_max)
    for i in np.nonzero(confident)[0]:
        rad = clear_nsigmas * max(float(sigma[i]), 1.0) + margin
        out.setdefault(int(idx[i]), []).append((float(r[i]), float(c[i]), rad))
    return out


def build_mask_file(
    inputs: list[str | Path],
    output: str | Path,
    *,
    peaks: list[str | Path] | None = None,
    peak_deviance_min: float = 25.0,
    peak_residual_max: float = 2.0,
    peak_clear_nsigmas: float = 3.0,
    min_frames: int = 5,
    smooth_sigma: float = 2.0,
    grad_nmads: float = 8.0,
    texture_factor: float = 0.25,
    wide_sigma: float = 10.0,
    dilate_px: int = 8,
    static_quantile: float = 25.0,
    grad_min_frac: float = 0.02,
) -> dict:
    """Estimate one mask per physical bank across every input file.

    Inputs are reduced/merged stacks (``images`` + ``bank_ids``); they may
    come from different samples.  The cleanest input is a control experiment
    without a sample, where everything is static and nothing needs rescuing.
    When only sample scans exist, ``peaks`` (finder outputs from an unmasked
    run, paired with ``inputs`` by order) exonerates detections whose fit
    metrics certify them as genuine: their footprints leave the static
    evidence per frame, so a reflection cannot be declared static however
    many frames it persists through -- which quasi-static Laue zones from a
    manually oriented crystal otherwise are.  A bank with fewer than
    ``min_frames`` frames gets a fully valid mask -- stated in the summary
    rather than silently guessed from statistics too thin to tell a peak
    from a shadow.
    """
    if peaks is not None and len(peaks) != len(inputs):
        raise ValueError(
            f"--peaks must pair with the inputs by order: got {len(peaks)} "
            f"peaks file(s) for {len(inputs)} input(s)"
        )
    by_bank, duplicates, provenance = _read_frames_by_bank([Path(p) for p in inputs])
    if not by_bank:
        raise ValueError("no frames found in the inputs")

    confident: dict[int, dict[int, list]] = {}
    n_exonerated = 0
    if peaks is not None:
        for file_index, peaks_path in enumerate(peaks):
            confident[file_index] = _confident_peaks_by_frame(
                peaks_path,
                peak_deviance_min,
                peak_residual_max,
                peak_clear_nsigmas,
                margin=2.0 * smooth_sigma,
            )

    banks = sorted(by_bank)
    masks, thin = [], []
    for bank in banks:
        frames = np.stack(by_bank[bank])
        if frames.shape[0] < min_frames:
            masks.append(np.ones(frames.shape[1:], dtype=np.uint8))
            thin.append(bank)
        else:
            clear_disks = None
            if confident:
                clear_disks = []
                for file_index, image_index in provenance[bank]:
                    disks = confident.get(file_index, {}).get(image_index, [])
                    n_exonerated += len(disks)
                    clear_disks.append(disks)
            masks.append(
                estimate_static_mask(
                    frames,
                    clear_disks=clear_disks,
                    smooth_sigma=smooth_sigma,
                    grad_nmads=grad_nmads,
                    texture_factor=texture_factor,
                    wide_sigma=wide_sigma,
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
        f.attrs["texture_factor"] = texture_factor
        f.attrs["wide_sigma"] = wide_sigma
        f.attrs["dilate_px"] = dilate_px
        f.attrs["static_quantile"] = static_quantile
        f.attrs["grad_min_frac"] = grad_min_frac

    masked_frac = 1.0 - stack.mean()
    return {
        "banks": banks,
        "n_frames": {b: len(by_bank[b]) for b in banks},
        "thin_banks": thin,
        "n_exonerated": n_exonerated,
        "duplicates_dropped": duplicates,
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
