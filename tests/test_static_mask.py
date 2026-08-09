"""Static-structure masks: estimation, bank mapping, and finder semantics.

The finder test is the contract that matters: masked pixels are missing
data, not zeroed counts, so detections vanish from the masked region while
a real peak elsewhere in the frame keeps its position and flux.
"""

from __future__ import annotations

import h5py
import numpy as np
import pytest
from scipy.special import erf

from subhkl.search.matrix_free import MatrixFreeSparseRBFPeakFinder
from subhkl.search.static_mask import (
    build_mask_file,
    estimate_static_mask,
    load_mask_for_banks,
)


def pixel_integrated_gaussian(shape, r0, c0, sigma, amplitude):
    rr, cc = np.mgrid[0 : shape[0], 0 : shape[1]].astype(float)
    s2 = sigma * np.sqrt(2.0)
    er = erf((rr - r0 + 0.5) / s2) - erf((rr - r0 - 0.5) / s2)
    ec = erf((cc - c0 + 0.5) / s2) - erf((cc - c0 - 0.5) / s2)
    return amplitude * (np.pi / 2.0) * sigma**2 * er * ec


def synthetic_scan(n_frames=12, size=128, step_col=80, glow=None, peaks=True, seed=5):
    """Frames with a static illumination step, optional static glow, and a
    bright peak that moves frame to frame -- the l1-mbl artifact anatomy."""
    rng = np.random.default_rng(seed)
    frames = []
    for k in range(n_frames):
        rate = np.full((size, size), 2.2)
        rate[:, step_col:] = 0.6  # illumination boundary, static
        if glow is not None:
            rr, cc = np.mgrid[0:size, 0:size]
            r0, c0, rad, level = glow
            rate += level * ((rr - r0) ** 2 + (cc - c0) ** 2 < rad**2)
        if peaks:
            rate += pixel_integrated_gaussian(
                rate.shape, 20.0 + 7 * k, 15.0 + 4 * k, 2.0, 80.0
            )
        frames.append(rng.poisson(rate).astype(np.float32))
    return np.stack(frames)


def test_static_mask_catches_the_boundary_and_glow_but_not_the_moving_peak():
    frames = synthetic_scan(glow=(100, 30, 10, 2.5))
    valid = estimate_static_mask(frames, dilate_px=4)

    # The illumination step is masked...
    assert valid[:, 76:84].mean() < 0.1
    # ...and so is the static glow.
    assert valid[92:108, 22:38].mean() < 0.2
    # The moving peak's positions stay valid: the median never saw it.
    for k in range(12):
        assert valid[20 + 7 * k, 15 + 4 * k] == 1
    # And the bulk of the panel is untouched.
    assert valid.mean() > 0.7


def test_static_mask_needs_at_least_two_frames():
    with pytest.raises(ValueError):
        estimate_static_mask(np.zeros((1, 32, 32)))


def test_mask_file_round_trip_maps_by_physical_bank(tmp_path):
    # Moving peaks keep the frames genuinely distinct scenes: peak-free
    # re-draws of one static pattern are re-exposures, and the duplicate
    # guard rightly collapses those.
    frames_a = synthetic_scan(n_frames=6, size=64, step_col=40, seed=1)
    frames_b = synthetic_scan(n_frames=6, size=64, step_col=20, seed=2)

    # Two input files, interleaved banks, on purpose out of order.
    for name, banks, stacks in (
        ("scan1.h5", [7, 3], (frames_a[:3], frames_b[:3])),
        ("scan2.h5", [3, 7], (frames_b[3:], frames_a[3:])),
    ):
        with h5py.File(tmp_path / name, "w") as f:
            f["images"] = np.concatenate(stacks)
            f["bank_ids"] = np.repeat(banks, 3)

    out = tmp_path / "mask.h5"
    summary = build_mask_file(
        [tmp_path / "scan1.h5", tmp_path / "scan2.h5"], out, min_frames=5, dilate_px=2
    )
    assert summary["banks"] == [3, 7]
    assert summary["n_frames"] == {3: 6, 7: 6}
    assert summary["thin_banks"] == []

    # Bank 3's step is at column 20, bank 7's at column 40; the mapping must
    # place each mask with its own bank whatever the requested order, and a
    # bank the file does not carry comes back fully valid.  On this small
    # fixture the strip is soft (sharpness is covered by the 128 px tests),
    # so the assertion is the mapping: each bank is masked at *its own*
    # step's bright shoulder and clean at the other bank's.
    stack = load_mask_for_banks(out, [7, 99, 3], (64, 64))
    assert stack.shape == (3, 64, 64)
    assert stack[0][:, 32:40].mean() < 0.5 and stack[0][:, 12:20].mean() > 0.9
    assert stack[2][:, 12:20].mean() < 0.5 and stack[2][:, 32:40].mean() > 0.9
    assert stack[1].min() == 1.0

    with pytest.raises(ValueError):
        load_mask_for_banks(out, [3], (32, 32))


def test_mask_file_keeps_thin_banks_fully_valid(tmp_path):
    frames = synthetic_scan(n_frames=3, size=64, peaks=False)
    with h5py.File(tmp_path / "scan.h5", "w") as f:
        f["images"] = frames
        f["bank_ids"] = np.full(3, 11)
    summary = build_mask_file(
        [tmp_path / "scan.h5"], tmp_path / "mask.h5", min_frames=5
    )
    assert summary["thin_banks"] == [11]
    assert load_mask_for_banks(tmp_path / "mask.h5", [11], (64, 64)).min() == 1.0


def test_finder_honors_the_mask_without_touching_the_counts():
    """An illumination step manufactures detections along its ridge; the mask
    removes them as missing data while the real peak keeps its flux."""
    rng = np.random.default_rng(9)
    size, step_col = 96, 60
    rate = np.full((size, size), 2.2)
    rate[:, step_col:] = 0.6
    rate += pixel_integrated_gaussian(rate.shape, 30.0, 25.0, 2.0, 120.0)
    image = rng.poisson(rate).astype(np.float32)
    stack = np.stack([image, image])

    def run(valid):
        finder = MatrixFreeSparseRBFPeakFinder(
            min_sigma=1.5,
            max_sigma=3.0,
            num_sigmas=3,
            profile_file="gaussian",
            shape_ratio=1.0,
        )
        return finder.find_peaks_batch(stack, valid=valid)

    unmasked = np.asarray(run(None)[0])

    valid = np.ones_like(stack)
    valid[:, :, step_col - 8 : step_col + 8] = 0.0
    masked = np.asarray(run(valid)[0])

    def near_step(peaks):
        return peaks[np.abs(peaks[:, 2] - step_col) < 8]

    def real_peak(peaks):
        sel = (np.abs(peaks[:, 1] - 30) < 3) & (np.abs(peaks[:, 2] - 25) < 3)
        return peaks[sel]

    # The step ridge attracts detections when unmasked -- that is the bug
    # being tested -- and none survive inside the mask.
    assert len(near_step(unmasked)) >= 1
    assert len(near_step(masked)) == 0

    # The real peak is found either way, at the same place and flux: masking
    # is missing data, not a modification of the counted image.
    a, b = real_peak(unmasked), real_peak(masked)
    assert len(a) == 1 and len(b) == 1
    np.testing.assert_allclose(a[0][1:3], b[0][1:3], atol=0.3)
    np.testing.assert_allclose(a[0][0], b[0][0], rtol=0.05)


def test_fully_valid_mask_is_the_unmasked_path():
    """A mask of ones must not even change the trace, let alone the answer."""
    rng = np.random.default_rng(3)
    image = rng.poisson(np.full((64, 64), 1.0)).astype(np.float32)
    image += pixel_integrated_gaussian(image.shape, 32.0, 32.0, 2.0, 100.0)
    stack = image[None]
    finder = MatrixFreeSparseRBFPeakFinder(
        min_sigma=1.5,
        max_sigma=3.0,
        num_sigmas=3,
        profile_file="gaussian",
        shape_ratio=1.0,
    )
    plain = np.asarray(finder.find_peaks_batch(stack)[0])
    ones = np.asarray(finder.find_peaks_batch(stack, valid=np.ones_like(stack))[0])
    np.testing.assert_array_equal(plain, ones)


def test_mask_is_invariant_to_exposure_time():
    """The same static feature at 5x the counts must give the same mask:
    frames are normalised to unit mean rate, and both criteria are ratios."""
    short = synthetic_scan(peaks=False, seed=4)
    long_ = np.stack(
        [
            np.random.default_rng(40 + k)
            .poisson(5.0 * (2.2 - 1.6 * (np.arange(128) >= 80)))
            .astype(np.float32)[None, :]
            .repeat(128, axis=0)
            for k in range(12)
        ]
    )
    # Rebuild the long exposure properly: same rate pattern as synthetic_scan,
    # scaled 5x, fresh Poisson draws.
    rng = np.random.default_rng(21)
    rate = np.full((128, 128), 2.2)
    rate[:, 80:] = 0.6
    long_ = np.stack([rng.poisson(5.0 * rate).astype(np.float32) for _ in range(12)])

    a = estimate_static_mask(short, dilate_px=4)
    b = estimate_static_mask(long_, dilate_px=4)
    # Same masked geometry.  The masked set is a ~20-column strip, so a
    # couple of columns of edge wiggle between independent noise draws
    # costs ~10-15% of Jaccard overlap; 0.75 asserts same-geometry while
    # tolerating that wiggle.
    both, either = ((a == 0) & (b == 0)).sum(), ((a == 0) | (b == 0)).sum()
    assert both / either > 0.75

    # And mixing the two exposures in one stack does not change the answer.
    mixed = estimate_static_mask(np.concatenate([short, long_]), dilate_px=4)
    both, either = ((mixed == 0) & (a == 0)).sum(), ((mixed == 0) | (a == 0)).sum()
    assert both / either > 0.75


def test_dense_diffraction_does_not_leak_into_the_static_map():
    """A reflection ring that keeps some peak near the same pixels in half
    the frames polluted a median-based map and masked genuine Bragg peaks
    (measured on l1-mbl forward banks).  The low-quantile map is clean."""
    rng = np.random.default_rng(17)
    size = 128
    frames = []
    for k in range(12):
        rate = np.full((size, size), 1.0)
        rate[:, 90:] = 0.3  # keep a real static edge in the frame
        # Dense pattern: bright peaks on a ring, positions jittering a few
        # pixels frame to frame.  Each position is lit in 2 of 12 frames --
        # the physical rate for a reflection in a rotation scan -- so the
        # ring as a whole is always occupied somewhere while no pixel is.
        for j in range(10):
            if (k + j) % 6:
                continue
            angle = 2 * np.pi * j / 10
            r0 = 60 + 25 * np.sin(angle) + rng.integers(-2, 3)
            c0 = 45 + 25 * np.cos(angle) + rng.integers(-2, 3)
            rate += pixel_integrated_gaussian(rate.shape, r0, c0, 2.0, 60.0)
        frames.append(rng.poisson(rate).astype(np.float32))
    valid = estimate_static_mask(np.stack(frames), dilate_px=4)

    # The static edge's bright shoulder -- where the background model
    # overshoots and false atoms form -- is masked.  The dark side is
    # deliberately not: false peaks are positive structure.
    assert valid[:, 84:90].mean() < 0.2
    rr, cc = np.mgrid[0:size, 0:size]
    ring = np.abs(np.hypot(rr - 60.0, cc - 45.0) - 25.0) < 6
    assert valid[ring].mean() > 0.9


def test_a_wide_smooth_halo_is_not_masked_and_its_peak_survives():
    """The background model follows broad structure, so the mask must not
    claim it: absolute-level and absolute-gradient criteria removed genuine
    Bragg peaks from the centre of the l1-mbl forward banks."""
    rng = np.random.default_rng(23)
    size = 128
    rr, cc = np.mgrid[0:size, 0:size]
    halo = 3.0 * np.exp(-((rr - 64.0) ** 2 + (cc - 64.0) ** 2) / (2 * 40.0**2))
    frames = []
    for k in range(12):
        rate = 1.0 + halo
        rate[:, 100:] = 0.3  # a real step stays in frame for contrast
        frames.append(rng.poisson(rate).astype(np.float32))
    valid = estimate_static_mask(np.stack(frames), dilate_px=4)

    # The halo centre -- 4x ambient, statically steep shoulders -- is valid.
    assert valid[44:84, 44:84].mean() > 0.95
    # The genuine step is still masked.
    assert valid[:, 96:104].mean() < 0.2


def test_repeated_frames_do_not_promote_signal_into_the_mask(tmp_path):
    """The same frame (or the same goniometer orientation re-exposed) fed
    repeatedly would make the true signal static by construction.  Repeats
    are dropped, reported, and a bank left with too few distinct frames
    stays fully valid."""
    rng = np.random.default_rng(31)
    rate = np.full((64, 64), 1.0)
    rate += pixel_integrated_gaussian(rate.shape, 30.0, 30.0, 2.0, 200.0)
    frame = rng.poisson(rate).astype(np.float32)

    # Same content ten times over (no goniometer bookkeeping in the file).
    with h5py.File(tmp_path / "dup.h5", "w") as f:
        f["images"] = np.stack([frame] * 10)
        f["bank_ids"] = np.full(10, 5)
    summary = build_mask_file([tmp_path / "dup.h5"], tmp_path / "m1.h5", min_frames=5)
    assert summary["duplicates_dropped"] == {5: 9}
    assert summary["thin_banks"] == [5]
    assert load_mask_for_banks(tmp_path / "m1.h5", [5], (64, 64)).min() == 1.0

    # Re-exposures: same angles, fresh Poisson draws.  Content differs, the
    # orientation does not, and the goniometer record is what catches it.
    with h5py.File(tmp_path / "reexp.h5", "w") as f:
        f["images"] = np.stack(
            [rng.poisson(rate).astype(np.float32) for _ in range(10)]
        )
        f["bank_ids"] = np.full(10, 5)
        f["goniometer/angles"] = np.tile([10.0, 0.0, 0.0], (10, 1))
    summary = build_mask_file([tmp_path / "reexp.h5"], tmp_path / "m2.h5", min_frames=5)
    assert summary["duplicates_dropped"] == {5: 9}
    assert load_mask_for_banks(tmp_path / "m2.h5", [5], (64, 64)).min() == 1.0


def test_no_exclusion_ring_around_a_static_hot_spot():
    """A static bright spot must mask its own footprint and nothing more:
    the |band-pass| criterion dug a ~wide-sigma moat around anything bright,
    which on real forward banks became a ~20 px exclusion zone around every
    leaked reflection."""
    rng = np.random.default_rng(41)
    size = 128
    frames = []
    for _ in range(12):
        rate = np.full((size, size), 1.0)
        rate += pixel_integrated_gaussian(rate.shape, 64.0, 64.0, 2.5, 50.0)
        frames.append(rng.poisson(rate).astype(np.float32))
    valid = estimate_static_mask(np.stack(frames), dilate_px=4)

    rr, cc = np.mgrid[0:size, 0:size]
    radius = np.hypot(rr - 64.0, cc - 64.0)
    # Core masked (it is static structure), annulus untouched.
    assert valid[radius < 4].mean() < 0.2
    assert valid[(radius > 16) & (radius < 28)].mean() > 0.95
