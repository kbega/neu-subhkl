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
    texture_factor: float = 0.15,
    wide_sigma: float = 10.0,
    dilate_px: int = 8,
    static_quantile: float = 25.0,
    grad_min_frac: float = 0.02,
    clear_disks: list | None = None,
    clear_nsigmas: float = 3.5,
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

    import warnings as _warnings

    def _reduce(radii: list | None) -> tuple[np.ndarray, np.ndarray]:
        """(static map, evidence validity) with per-disk evidence cleared."""
        work = stack
        if radii is not None:
            work = stack.copy()
            for fi, disks in enumerate(clear_disks):
                for (r0, c0, _sig, _amp), rad in zip(disks, radii[fi]):
                    work[fi][(rr - r0) ** 2 + (cc - c0) ** 2 <= rad**2] = np.nan
        with _warnings.catch_warnings():
            _warnings.simplefilter("ignore", category=RuntimeWarning)
            sm = np.nanpercentile(work, static_quantile, axis=0, method="lower")
        # A pixel with too little surviving evidence proves nothing static.
        n_eff = np.sum(~np.isnan(work), axis=0)
        valid = (n_eff >= 2) & np.isfinite(sm)
        return np.where(valid, sm, 0.0), valid

    def _threshold(
        sm: np.ndarray, valid: np.ndarray
    ) -> tuple[float, np.ndarray, float]:
        """(ambient, band-pass map, texture threshold) for a static map.

        The criterion lives on a *band-pass* of the static map: structure at
        scales between the atom footprint (smooth_sigma) and the background
        window (wide_sigma).  What the mask must mark is structure that
        defeats the background model -- and only that.  A wide static halo,
        however elevated or statically steep its shoulders, is followed by
        the windowed background estimate and is where real Bragg peaks live;
        masking it (as an absolute-gradient or absolute-level criterion does)
        removed genuine reflections from the centre of the l1-mbl forward
        banks.  The halo vanishes from the band-pass; what remains is exactly
        the peak-confusable statics: the illumination step (band-pass swing
        ~0.4x ambient), the plume's texture (~0.3x at its false-detection
        sites, against ~0.13x under real forward-bank peaks).

        The wide smooth is a *normalised convolution*: cleared pixels carry
        zero weight, not zero value.  A quasi-static certified peak clears
        an overlapping disk in every frame, so the static map has a
        no-evidence crater there; averaging the crater's zeros into the
        wide background dug that background down around it, and the
        band-pass acquired a positive rim just outside the crater -- up to
        the crater's share of the wide kernel times ambient, comfortably
        above any texture threshold.  The rim was then masked: a thin
        annulus around every certified bright peak, manufactured by the
        clearing itself.  With the evidence-weighted average the background
        estimate outside the crater uses only real evidence, and inside it
        the band is pinned to zero -- no evidence, no criteria.

        The threshold is the *larger* of the effect-size criterion
        (texture_factor x ambient) and a significance floor on the band's own
        noise.  The boundary criterion always had its MAD test; the texture
        criterion did not, so on a photon-sparse panel an aggressive
        texture_factor would start masking Poisson speckle.  4 MADs is ~2.7
        sigma of the band's own noise: measured on photon-sparse fixtures
        (rate ~1-2, few frames) it sits at 0.19-0.26x ambient -- where the
        old fixed threshold was -- while real detector panels, with more
        frames and area behind the order statistic, have MADs far below any
        sensible texture_factor, so there the factor alone governs.
        """
        vals = sm[valid] if valid.any() else sm.ravel()
        ambient = max(float(np.median(vals)), 1e-3)
        weight = valid.astype(np.float32)
        num = ndimage.gaussian_filter(sm * weight, wide_sigma)
        den = ndimage.gaussian_filter(weight, wide_sigma)
        wide = np.where(den > 1e-3, num / np.maximum(den, 1e-3), 0.0)
        band = np.where(valid, sm - wide, 0.0)
        bvals = band[valid] if valid.any() else band.ravel()
        band_mad = np.median(np.abs(bvals - np.median(bvals))) + 1e-9
        return ambient, band, max(texture_factor * ambient, 4.0 * band_mad)

    def _tail_radius(sig: float, amp: float, scale: float, thr: float) -> float:
        """Radius where the peak's smoothed profile falls below ``thr``.

        Smoothing spreads the peak over s_eff and attenuates its amplitude
        by the area ratio; both in normalised rate units.
        """
        s_eff_sq = sig**2 + smooth_sigma**2
        amp_rel = (amp / max(scale, 1e-6)) * (sig**2 / s_eff_sq)
        if amp_rel > thr:
            return float(np.sqrt(2.0 * s_eff_sq * np.log(amp_rel / thr)))
        return 0.0

    # Exonerated peaks leave the evidence pool: ``clear_disks`` carries, per
    # frame, (row, col, sigma, amplitude) of detections whose fit metrics
    # certify them as genuine (see build_mask_file).  Removing the footprint
    # from *this frame's* evidence -- rather than subtracting a model, which
    # would leave mismatch dipoles -- means a real reflection cannot be
    # declared static however many frames it persists through, which is
    # exactly what happens when a manually oriented crystal repeats its
    # Laue zones.
    #
    # The clearing radius must know the peak's brightness, like the final
    # protection below: a fixed n-sigma disk leaves a bright peak's tail
    # above the texture threshold in every frame, and a quasi-static peak
    # (a Laue zone that barely moves under rotation about its own axis)
    # then writes an annulus of "static" texture into the map that the
    # protection had to claw back -- imperfectly, because the aggregate is
    # broadened by the few-pixel wobble beyond the fitted model.  But the
    # amplitude-aware radius needs the texture threshold, which is measured
    # *from* the cleared map -- so the reduction runs twice: a blind pass
    # sets the threshold, the aware pass removes the tails at their source.
    # A 2*smooth_sigma margin on the tail crossing absorbs the wobble
    # broadening the fitted sigma does not know about.
    rr = cc = None
    radii = None
    if clear_disks is not None:
        rr, cc = np.mgrid[0 : stack.shape[1], 0 : stack.shape[2]]
        radii = [
            [clear_nsigmas * sig + 2.0 * smooth_sigma for _r, _c, sig, _a in disks]
            for disks in clear_disks
        ]
        smooth, valid = _reduce(radii)
        _, _, thr0 = _threshold(smooth, valid)
        aware = [
            [
                max(
                    blind,
                    _tail_radius(sig, amp, float(scales[fi].squeeze()), thr0)
                    + 2.0 * smooth_sigma,
                )
                for (_r, _c, sig, amp), blind in zip(disks, radii[fi])
            ]
            for fi, disks in enumerate(clear_disks)
        ]
        grew = any(a > b + 0.5 for fa, fb in zip(aware, radii) for a, b in zip(fa, fb))
        radii = aware
        if grew:
            smooth, valid = _reduce(radii)
    else:
        smooth, valid = _reduce(None)

    ambient, band, texture_threshold = _threshold(smooth, valid)
    # Positive lobes only.  False peaks are *positive* unmodelled structure;
    # |band| would additionally mask the negative moat that the wide
    # subtraction digs around anything bright -- a ~wide_sigma-to-2*wide_sigma
    # ring, which on the forward banks turned every leak into a ~20 px
    # exclusion zone around a genuine reflection.
    texture = band > texture_threshold

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

    # Exoneration must also *protect the final mask*.  The evidence clearing
    # above removes the tails at their source, but a peak certified in only
    # some frames still leaks through the others' quantile, and the pass-1
    # threshold the clearing radii were computed against can sit slightly
    # above the final one.  The protected radius is the cleared radius plus
    # the dilation the bad set just received, recomputed against the final
    # threshold, and it is subtracted *after* dilation so nothing grows
    # back in.  A certified reflection's footprint is then never masked,
    # however bright.
    if clear_disks is not None:
        protected = np.zeros_like(bad)
        for fi, disks in enumerate(clear_disks):
            scale = float(scales[fi].squeeze())
            for (r0, c0, sig, amp), cleared in zip(disks, radii[fi]):
                r_tail = _tail_radius(sig, amp, scale, texture_threshold)
                rad = max(cleared, r_tail + 2.0 * smooth_sigma) + dilate_px
                protected |= (rr - r0) ** 2 + (cc - c0) ** 2 <= rad**2
        bad &= ~protected

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


def _first_instrument(inputs: list) -> str | None:
    """The instrument the first input that knows one records."""
    for path in inputs:
        with h5py.File(path, "r") as f_in:
            recorded = f_in.attrs.get("instrument")
        if isinstance(recorded, bytes):
            recorded = recorded.decode("utf-8")
        if recorded:
            return recorded
    return None


def build_summed_file(inputs: list[str | Path], output: str | Path) -> dict:
    """One frame per physical bank: the sum of its deduplicated frames.

    The companion input to the exoneration: a finder run on this stack sees
    the *pooled* evidence, exactly as the static-map quantile does.  Per-peak
    deviance is additive across frames, so a quasi-static reflection sitting
    just below any single frame's admission level compounds to certification
    here (~n_frames-fold in deviance), and the pooled fit's width knows the
    frame-to-frame wobble the single-frame fits cannot.  Feed the resulting
    peaks file to ``build_mask_file`` as ``pooled_peaks``.

    Frames pass through the same per-bank deduplication as the mask
    estimator, so the pooled statistics describe exactly the evidence the
    mask is built from.  The structural metadata a finder needs
    (instrument/wavelength, sample, goniometer axes) is copied from the
    first input; per-frame goniometer angles are meaningless for a sum and
    are written as zeros, so this file is for peak *metrics* only -- never
    index from it.
    """
    paths = [Path(p) for p in inputs]
    by_bank, duplicates, _ = _read_frames_by_bank(paths)
    if not by_bank:
        raise ValueError("no frames found in the inputs")
    banks = sorted(by_bank)
    sums = np.stack([np.sum(by_bank[b], axis=0) for b in banks])
    n_frames = [len(by_bank[b]) for b in banks]

    instrument = _first_instrument(paths)
    with h5py.File(output, "w") as f:
        f.create_dataset("images", data=sums, compression="gzip", compression_opts=4)
        f["bank_ids"] = np.asarray(banks, dtype=np.int64)
        f.attrs["kind"] = "summed-frames"
        f.attrs["inputs"] = [str(p) for p in paths]
        f.attrs["n_frames"] = np.asarray(n_frames, dtype=np.int64)
        if instrument:
            f.attrs["instrument"] = instrument
        with h5py.File(paths[0], "r") as f_in:
            for key in (
                "instrument/wavelength",
                "sample",
                "goniometer/axes",
                "goniometer/names",
            ):
                if key in f_in:
                    group, _, leaf = key.rpartition("/")
                    dest = f.require_group(group) if group else f
                    f_in.copy(key, dest, name=leaf)
            if "goniometer/angles" in f_in:
                width = f_in["goniometer/angles"].shape[1:]
                f["goniometer/angles"] = np.zeros((len(banks), *width))
    return {
        "banks": banks,
        "n_frames": dict(zip(banks, n_frames)),
        "duplicates_dropped": duplicates,
        "output": str(output),
    }


# chi^2_4 at 95%: the admission level for a four-parameter atom.  The
# exoneration bar sits here, not above it: the finder's reported peaks
# already passed a *calibrated* panel-wide false-alarm control (E[FP] per
# bank), so a higher deviance bar re-litigates evidence the calibration
# settled -- and manufactures a gap.  Masking a quasi-static peak takes
# only ~texture_factor x ambient in the smoothed p25 (~15-20 recorded
# counts), while a deviance-25 certificate takes ~3x that; measured on
# l1-mbl, 340 detections fell in between -- every one with residual
# deviance per DoF < 2 (median 1.11), deviance 20-23, median flux 133 --
# faint genuine reflections masked with no route to exoneration.  84% sat
# in compact blobs the size of their own dilated footprint, not on any
# extended static structure.
CHI2_4_P95 = 9.488


def _confident_peaks_by_frame(
    peaks_path: str | Path,
    deviance_min: float,
    residual_max: float,
) -> dict[int, list[tuple[float, float, float, float]]]:
    """Per image index: (row, col, sigma, amplitude) of certified peaks.

    The certificate is the finder's own fit statistics: evidence at least
    at the chi^2_4 admission level (the finder's calibrated false-alarm
    control already governs above it -- see CHI2_4_P95) and a shape the
    atom family explains (residual deviance per DoF near one).  An
    artifact fails one or both and earns no exoneration.  The geometry of
    the cleared/protected region is the estimator's business -- it knows
    the ambient rate a peak's tail must be compared against.
    """
    out: dict[int, list[tuple[float, float, float, float]]] = {}
    with h5py.File(peaks_path, "r") as f:
        idx = np.asarray(f["peaks/image_index"][()]).astype(int)
        r = np.asarray(f["peaks/pixel_r"][()], dtype=float)
        c = np.asarray(f["peaks/pixel_c"][()], dtype=float)
        sigma = np.asarray(f["peaks/sigma"][()], dtype=float)
        flux = np.asarray(f["peaks/intensity"][()], dtype=float)
        dev = np.asarray(f["peaks/deviance"][()], dtype=float)
        res = np.asarray(f["peaks/residual_deviance"][()], dtype=float)
    confident = (dev > deviance_min) & (res < residual_max)
    for i in np.nonzero(confident)[0]:
        s_i = max(float(sigma[i]), 1.0)
        amp = max(float(flux[i]), 0.0) / (2.0 * np.pi * s_i**2)
        out.setdefault(int(idx[i]), []).append((float(r[i]), float(c[i]), s_i, amp))
    return out


def _confident_pooled_peaks(
    peaks_path: str | Path,
    deviance_min: float,
    residual_max: float,
) -> dict[int, list[tuple[float, float, float, float]]]:
    """Per physical bank: certified peaks of a finder run on summed frames.

    Per-frame certification cannot see what the mask sees: the static map
    pools ~n frames of evidence, so a quasi-static reflection just below
    every single frame's admission level is masked with no route to
    exoneration.  A finder run on the per-bank *sum* (see
    ``build_summed_file``) closes the asymmetry -- deviance is additive, so
    the same peak compounds ~n-fold there -- and its certified detections
    are exonerated in every frame of their bank.  The amplitude recorded
    here is the pooled one; the caller rescales by the bank's frame count
    to recover the per-frame amplitude (exact for a static feature, an
    overestimate only for moving peaks, which the quantile never masks and
    which therefore cost nothing to over-protect).

    The certificate is the same as per-frame; the bank comes from the
    finder's own per-peak ``bank`` dataset, so the pooled file needs no
    companion to interpret.
    """
    out: dict[int, list[tuple[float, float, float, float]]] = {}
    with h5py.File(peaks_path, "r") as f:
        if "bank" not in f:
            raise ValueError(
                f"{peaks_path}: pooled peaks need a per-peak 'bank' dataset "
                "(a finder output has one)"
            )
        bank = np.asarray(f["bank"][()]).astype(int)
        r = np.asarray(f["peaks/pixel_r"][()], dtype=float)
        c = np.asarray(f["peaks/pixel_c"][()], dtype=float)
        sigma = np.asarray(f["peaks/sigma"][()], dtype=float)
        flux = np.asarray(f["peaks/intensity"][()], dtype=float)
        dev = np.asarray(f["peaks/deviance"][()], dtype=float)
        res = np.asarray(f["peaks/residual_deviance"][()], dtype=float)
    confident = (dev > deviance_min) & (res < residual_max)
    for i in np.nonzero(confident)[0]:
        s_i = max(float(sigma[i]), 1.0)
        amp = max(float(flux[i]), 0.0) / (2.0 * np.pi * s_i**2)
        out.setdefault(int(bank[i]), []).append((float(r[i]), float(c[i]), s_i, amp))
    return out


def build_mask_file(
    inputs: list[str | Path],
    output: str | Path,
    *,
    peaks: list[str | Path] | None = None,
    pooled_peaks: str | Path | None = None,
    peak_deviance_min: float = CHI2_4_P95,
    peak_residual_max: float = 2.0,
    peak_clear_nsigmas: float = 3.5,
    min_frames: int = 5,
    smooth_sigma: float = 2.0,
    grad_nmads: float = 8.0,
    texture_factor: float = 0.15,
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
    manually oriented crystal otherwise are.  ``pooled_peaks`` (a finder
    output from the per-bank *summed* stack, see ``build_summed_file``)
    extends the same rescue to reflections too faint for any single frame's
    certificate: their significance compounds across frames there, and a
    certified pooled peak is exonerated in every frame of its bank.  A bank
    with fewer than ``min_frames`` frames gets a fully valid mask -- stated
    in the summary rather than silently guessed from statistics too thin to
    tell a peak from a shadow.
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
            )
    pooled: dict[int, list] = {}
    n_exonerated_pooled = 0
    if pooled_peaks is not None:
        pooled = _confident_pooled_peaks(
            pooled_peaks,
            peak_deviance_min,
            peak_residual_max,
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
            bank_pooled = [
                # The pooled amplitude is a sum over this bank's frames; a
                # static feature's per-frame amplitude is that divided by
                # the frame count (exact by definition of static).
                (r0, c0, sig, amp / frames.shape[0])
                for r0, c0, sig, amp in pooled.get(bank, [])
            ]
            n_exonerated_pooled += len(bank_pooled)
            if confident or bank_pooled:
                clear_disks = []
                for file_index, image_index in provenance[bank]:
                    disks = confident.get(file_index, {}).get(image_index, [])
                    n_exonerated += len(disks)
                    clear_disks.append(disks + bank_pooled)
            masks.append(
                estimate_static_mask(
                    frames,
                    clear_disks=clear_disks,
                    clear_nsigmas=peak_clear_nsigmas,
                    smooth_sigma=smooth_sigma,
                    grad_nmads=grad_nmads,
                    texture_factor=texture_factor,
                    wide_sigma=wide_sigma,
                    dilate_px=dilate_px,
                    static_quantile=static_quantile,
                    grad_min_frac=grad_min_frac,
                )
            )

    instrument = _first_instrument(inputs)

    stack = np.stack(masks)
    with h5py.File(output, "w") as f:
        # Mostly-ones uint8 compresses ~100x, and the benchmark harness ships
        # this file in its CI artifact.
        f.create_dataset("images", data=stack, compression="gzip", compression_opts=4)
        f["bank_ids"] = np.asarray(banks, dtype=np.int64)
        f.attrs["kind"] = "static-mask"
        if instrument:
            f.attrs["instrument"] = instrument
        f.attrs["inputs"] = [str(p) for p in inputs]
        f.attrs["min_frames"] = min_frames
        f.attrs["smooth_sigma"] = smooth_sigma
        f.attrs["grad_nmads"] = grad_nmads
        f.attrs["texture_factor"] = texture_factor
        f.attrs["wide_sigma"] = wide_sigma
        f.attrs["dilate_px"] = dilate_px
        f.attrs["static_quantile"] = static_quantile
        f.attrs["grad_min_frac"] = grad_min_frac
        # The exoneration provenance: without it, a mask file cannot answer
        # "were peaks passed, and at what bar?" -- the first question asked
        # when a peak turns up masked.
        f.attrs["peaks"] = [str(p) for p in peaks] if peaks else []
        f.attrs["pooled_peaks"] = str(pooled_peaks) if pooled_peaks else ""
        f.attrs["peak_deviance_min"] = peak_deviance_min
        f.attrs["peak_residual_max"] = peak_residual_max
        f.attrs["peak_clear_nsigmas"] = peak_clear_nsigmas
        f.attrs["n_exonerated"] = n_exonerated
        f.attrs["n_exonerated_pooled"] = n_exonerated_pooled

    masked_frac = 1.0 - stack.mean()
    return {
        "banks": banks,
        "n_frames": {b: len(by_bank[b]) for b in banks},
        "thin_banks": thin,
        "n_exonerated": n_exonerated,
        "n_exonerated_pooled": n_exonerated_pooled,
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
