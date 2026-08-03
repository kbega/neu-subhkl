"""Sparse-RBF finder and integrator regression tests.

All peak-*finding* cases here run at ``gamma=0.5``.  ``gamma=1`` must not be
used: it is the point at which the penalty per unit flux becomes independent of
scale, so one broad atom and a mass-preserving spread of narrower ones have the
same cost and the same predicted image.  The minimiser is then not unique in the
scale coordinate, and because extra atoms always absorb a little more noise the
fit breaks the tie towards splitting -- a single peak is reported as a cluster.
See docs/matrix_free_theory.md, Theorem 1.

``gamma=0.5`` is used uniformly so that no test depends on its own tuning.  It
sits inside the usable range on both sides: ``gamma=1`` fragments, while
``gamma=0`` over-merges and swallows genuine neighbours.  A case needing a
different value should say why at its own call site.

Integrator call sites keep their original gamma deliberately.  The degeneracy
above requires unknown positions *and* unknown scale at once; integration is
handed positions from the lattice and performs no model selection over them, so
scale is identified there and the argument does not apply.
"""

import numpy as np
import scipy.special


def generate_erf_peak(y_coords, x_coords, r, c, sig, amp):
    """
    Helper function to generate physically exact subpixel peaks
    using the continuous analytic Gaussian pixel integral.
    """
    sig_sq2 = sig * np.sqrt(2.0) + 1e-6
    erf_y = scipy.special.erf((y_coords + 0.5 - r) / sig_sq2) - scipy.special.erf(
        (y_coords - 0.5 - r) / sig_sq2
    )
    erf_x = scipy.special.erf((x_coords + 0.5 - c) / sig_sq2) - scipy.special.erf(
        (x_coords - 0.5 - c) / sig_sq2
    )
    return amp * (np.pi / 2.0) * (sig**2) * erf_y * erf_x


def test_single_isolated_peak():
    """
    Validates that a single isolated Gaussian peak is correctly integrated by the
    new Patch-Based SSN Integrator, that the best shape is activated, and that
    the unpenalized Tikhonov debiasing accurately recovers the mass.
    """
    try:
        from subhkl.peakfinder.sparse_rbf import SparseLaueIntegrator
    except ImportError:
        from subhkl.search.sparse_rbf import SparseLaueIntegrator

    import numpy as np

    H, W = 50, 50
    bg_level = 15.0

    np.random.seed(42)
    y_coords, x_coords = np.meshgrid(np.arange(H), np.arange(W), indexing="ij")

    # Flat background
    image = np.full((H, W), bg_level, dtype=np.float32)

    cx, cy = 25.0, 25.0
    true_sigma = 2.0
    true_amp = 100.0

    # generate_erf_peak should already be in your test file
    image += generate_erf_peak(y_coords, x_coords, cy, cx, true_sigma, true_amp)

    # Apply Poisson noise
    image = np.random.poisson(image).astype(np.float32)

    # Use the new Unified Patch Integrator
    integrator = SparseLaueIntegrator(
        alpha=4.0,  # 4-sigma detection threshold
        min_sigma=1.0,
        max_sigma=5.0,
        # gamma=1 is safe here, unlike in the finder: integration is given the
        # peak positions and does no model selection over them, so the scale
        # degeneracy of docs/matrix_free_theory.md Theorem 1 does not apply.
        gamma=1.0,
        loss="gaussian",
    )

    images_batch = image[np.newaxis, ...]
    frames = [0]
    rs = [cy]
    cs = [cx]

    results = integrator.integrate_reflections(images_batch, frames, rs, cs)

    assert len(results) == 1, "The integrator dropped the peak!"

    intensity, r_found, c_found, sig_found, sigI_found = results[0]

    # 1. Did it pick the right shape from the linspace dictionary?
    assert abs(sig_found - true_sigma) < 0.25, (
        f"Sigma warped! Expected ~{true_sigma}, Found {sig_found}"
    )

    # 2. Did the debiasing properly recover the physical mass?
    expected_intensity = true_amp * 2 * np.pi * true_sigma**2
    assert np.isclose(intensity, expected_intensity, rtol=0.15), (
        f"Debiasing failed: {intensity} vs {expected_intensity}"
    )


def test_overlapping_peaks_crosstalk():
    """
    Validates that the patch-based integrator can independently resolve closely
    overlapping peaks without the backgrounds swallowing each other, thanks to the
    local median filter and robust NCC warm start.
    """
    try:
        from subhkl.peakfinder.sparse_rbf import SparseLaueIntegrator
    except ImportError:
        from subhkl.search.sparse_rbf import SparseLaueIntegrator

    import numpy as np

    H, W = 50, 50
    bg_level = 10.0
    np.random.seed(101)

    image = np.full((H, W), bg_level, dtype=np.float32)
    y_coords, x_coords = np.meshgrid(np.arange(H), np.arange(W), indexing="ij")

    # Peak 1
    cx1, cy1 = 20.0, 25.0
    true_sig1, true_amp1 = 2.0, 80.0
    image += generate_erf_peak(y_coords, x_coords, cy1, cx1, true_sig1, true_amp1)

    # Peak 2 (Highly overlapping, only 2-sigma away!)
    cx2, cy2 = 24.0, 25.0
    true_sig2, true_amp2 = 2.0, 60.0
    image += generate_erf_peak(y_coords, x_coords, cy2, cx2, true_sig2, true_amp2)

    image = np.random.poisson(image).astype(np.float32)

    integrator = SparseLaueIntegrator(
        alpha=4.0, min_sigma=1.0, max_sigma=5.0, gamma=2.0, loss="gaussian"
    )

    images_batch = image[np.newaxis, ...]
    frames = [0, 0]
    rs = [cy1, cy2]
    cs = [cx1, cx2]

    results = integrator.integrate_reflections(images_batch, frames, rs, cs)

    assert len(results) == 2, "Integrator crashed on one of the overlapping peaks!"

    i1, r1, c1, sig1, sigI1 = results[0]
    i2, r2, c2, sig2, sigI2 = results[1]

    # Ensure both survived the sparsity constraints
    assert sig1 > 0.0, "Peak 1 was crushed"
    assert sig2 > 0.0, "Peak 2 was crushed"

    # Because we evaluate them as independent local patches now (instead of a giant joint matrix),
    # there is a slight geometric overlap accepted into the unpenalized volume.
    # We use a 20% tolerance to ensure crosstalk bleeding stays mathematically bounded.
    exp_i1 = true_amp1 * 2 * np.pi * true_sig1**2
    exp_i2 = true_amp2 * 2 * np.pi * true_sig2**2

    assert np.isclose(i1, exp_i1, rtol=0.20), (
        f"Peak 1 Crosstalk Bleed: {i1} vs {exp_i1}"
    )
    assert np.isclose(i2, exp_i2, rtol=0.20), (
        f"Peak 2 Crosstalk Bleed: {i2} vs {exp_i2}"
    )


def test_integrate_peaks_rbf_ssn_orchestrator():
    H, W = 40, 40
    image = np.full((H, W), 5.0, dtype=np.float32)
    cx, cy = 20.0, 20.0
    y_coords, x_coords = np.meshgrid(np.arange(H), np.arange(W), indexing="ij")

    true_sigma = 2.0
    true_amp = 50.0
    image += generate_erf_peak(y_coords, x_coords, cy, cx, true_sigma, true_amp)

    class MockImageHandler:
        def __init__(self, ims):
            self.ims = ims
            self.bank_mapping = {}

        def get_run_id(self, img_key):
            return 0

    class MockPeaks:
        def __init__(self, ims):
            self.image = MockImageHandler(ims)
            self.config = {
                "0": {
                    "detector": {
                        "n": H,
                        "m": W,
                        "width": W * 1.0,
                        "height": H * 1.0,
                        "pixel_size": 1.0,
                        "center": [0.0, 0.0, 100.0],
                        "uhat": [1.0, 0.0, 0.0],
                        "vhat": [0.0, 1.0, 0.0],
                        "panel": "flat",
                    }
                }
            }

        def get_run_id(self, img_key):
            return self.image.get_run_id(img_key)

        def get_detector_by_img(self, img_key):
            from subhkl.instrument.detector import Detector

            return Detector(self.config["0"]["detector"])

    mock_peaks_obj = MockPeaks({0: image})

    peak_dict = {
        0: [
            np.array([cx]),
            np.array([cy]),
            np.array([1]),
            np.array([2]),
            np.array([3]),
            np.array([1.5]),
        ]
    }

    try:
        from subhkl.peakfinder.sparse_rbf import integrate_peaks_rbf_ssn
    except ImportError:
        from subhkl.search.sparse_rbf import integrate_peaks_rbf_ssn

    res = integrate_peaks_rbf_ssn(
        peak_dict=peak_dict,
        peaks_obj=mock_peaks_obj,
        sigmas=[1.0, 2.0, 3.0],
        alpha=0.5,
        gamma=2.0,
        show_progress=False,
    )

    assert len(res.intensity) == 1

    expected_intensity = true_amp * 2 * np.pi * (true_sigma**2)

    assert res.intensity[0] > 0
    assert np.isclose(res.intensity[0], expected_intensity, rtol=0.15)


def test_peak_finder_multiscale_subpixel_recovery():
    import numpy as np

    from subhkl.search.matrix_free import MatrixFreeSparseRBFPeakFinder

    H, W = 60, 60

    np.random.seed(42)
    bg_level = 50.0
    image = np.random.poisson(bg_level, size=(H, W)).astype(np.float32)

    y_coords, x_coords = np.meshgrid(np.arange(H), np.arange(W), indexing="ij")

    gt_c1, gt_r1 = 30.0, 30.0
    gt_sig1 = 4.0
    gt_amp1 = 200.0
    image += generate_erf_peak(y_coords, x_coords, gt_r1, gt_c1, gt_sig1, gt_amp1)

    gt_c2, gt_r2 = 33.74, 34.21
    gt_sig2 = 1.0
    gt_amp2 = 120.0
    image += generate_erf_peak(y_coords, x_coords, gt_r2, gt_c2, gt_sig2, gt_amp2)

    image_batch = image[np.newaxis, ...]

    finder = MatrixFreeSparseRBFPeakFinder(
        alpha=None, gamma=0.5, min_sigma=0.5, max_sigma=5.0, show_steps=False
    )

    results = finder.find_peaks_batch(image_batch)
    peaks = results[0]

    assert len(peaks) >= 2

    dists_to_broad = np.sqrt((peaks[:, 1] - gt_r1) ** 2 + (peaks[:, 2] - gt_c1) ** 2)
    broad_idx = np.argmin(dists_to_broad)
    broad_peak = peaks[broad_idx]

    dists_to_sharp = np.sqrt((peaks[:, 1] - gt_r2) ** 2 + (peaks[:, 2] - gt_c2) ** 2)
    sharp_idx = np.argmin(dists_to_sharp)
    sharp_peak = peaks[sharp_idx]

    assert broad_idx != sharp_idx

    assert np.isclose(broad_peak[1], gt_r1, atol=1.0)
    assert np.isclose(broad_peak[2], gt_c1, atol=1.0)
    assert broad_peak[3] > 2.0

    assert np.isclose(sharp_peak[1], gt_r2, atol=0.5)
    assert np.isclose(sharp_peak[2], gt_c2, atol=0.5)
    assert sharp_peak[3] < 2.0


def test_gaussian_loss_path_finds_peaks():
    """The Gaussian likelihood path must still detect, since it is the CLI default.

    This replaces a test that compared *recovered flux* between the two losses.
    The finder no longer promises amplitudes -- the pipeline reduces its output
    to (row, column) and intensity is measured later by the integrator -- so an
    assertion on flux was testing a quantity with no consumer, and it was the
    only thing keeping the debiasing phase alive.  What still needs covering is
    that ``loss="gaussian"`` runs and localises, which is what this asserts.
    """
    import numpy as np

    from subhkl.search.matrix_free import MatrixFreeSparseRBFPeakFinder

    H, W = 60, 60
    np.random.seed(101)

    truth = [(20.0, 20.0, 2.0, 400.0), (40.0, 42.0, 3.0, 500.0)]
    y_coords, x_coords = np.meshgrid(np.arange(H), np.arange(W), indexing="ij")
    image = np.full((H, W), 30.0, dtype=np.float32)
    for r, c, sig, amp in truth:
        image += generate_erf_peak(y_coords, x_coords, r, c, sig, amp)
    image = np.random.poisson(image).astype(np.float32)

    for loss in ("gaussian", "poisson"):
        finder = MatrixFreeSparseRBFPeakFinder(
            gamma=0.5,
            min_sigma=1.0,
            max_sigma=5.0,
            loss=loss,
            show_steps=False,
        )
        peaks = finder.find_peaks_batch(image[np.newaxis, ...])[0]
        assert len(peaks) >= 2, f"{loss} loss found {len(peaks)} peaks, expected 2"
        for r, c, _sig, _amp in truth:
            d = np.sqrt((peaks[:, 1] - r) ** 2 + (peaks[:, 2] - c) ** 2)
            assert d.min() < 1.0, (
                f"{loss} loss missed the peak at ({r}, {c}); nearest was {d.min():.2f} px"
            )


def test_poisson_overlapping_string():
    import numpy as np

    from subhkl.search.matrix_free import MatrixFreeSparseRBFPeakFinder

    H, W = 40, 80
    np.random.seed(123)
    bg_level = 20.0

    image = np.full((H, W), bg_level, dtype=np.float32)
    y_coords, x_coords = np.meshgrid(np.arange(H), np.arange(W), indexing="ij")

    true_peaks = [
        (20.0, 30.0, 1.0, 150.0),
        (20.0, 33.0, 1.0, 160.0),
        (20.0, 36.0, 1.0, 140.0),
        (20.0, 39.0, 1.0, 150.0),
    ]

    for r, c, sig, amp in true_peaks:
        image += generate_erf_peak(y_coords, x_coords, r, c, sig, amp)

    image = np.random.poisson(image).astype(np.float32)
    image_batch = image[np.newaxis, ...]

    finder = MatrixFreeSparseRBFPeakFinder(
        alpha=None,
        gamma=0.5,
        min_sigma=0.5,
        max_sigma=5.0,
        loss="poisson",
        show_steps=False,
    )

    results = finder.find_peaks_batch(image_batch)
    peaks = results[0]

    roi_mask = (
        (peaks[:, 1] > 15)
        & (peaks[:, 1] < 25)
        & (peaks[:, 2] > 25)
        & (peaks[:, 2] < 45)
    )
    roi_peaks = peaks[roi_mask]

    assert len(roi_peaks) >= 3

    # FIX: Evaluate deviance against the model's actual estimated background
    medians_ideal = np.array([bg_level])[np.newaxis, ...]
    bg_map = getattr(finder, "_last_bg_map", medians_ideal)

    metrics = finder.compute_metrics(image_batch, bg_map, [peaks], global_max=1.0)
    deviance = metrics["deviance_nu"]

    # Allow < 2.5 to account for L1 shrinkage bias on highly degenerate, overlapping strings
    assert deviance < 2.5, f"Deviance too high ({deviance:.2f})"


def test_real_neutron_structured_background():
    import numpy as np

    from subhkl.search.matrix_free import MatrixFreeSparseRBFPeakFinder

    H, W = 100, 100
    np.random.seed(42)

    y_coords, x_coords = np.meshgrid(np.arange(H), np.arange(W), indexing="ij")

    halo_amp = 80.0
    halo_sig = 30.0
    r2_halo = (x_coords - 50) ** 2 + (y_coords - 50) ** 2
    bg_structured = 15.0 + halo_amp * np.exp(-r2_halo / (2 * halo_sig**2))
    image = np.copy(bg_structured)

    true_peaks = [
        (25.0, 25.0, 1.2, 300.0),
        (75.0, 75.0, 1.0, 80.0),
        (50.0, 50.0, 2.0, 400.0),
    ]

    for r, c, sig, amp in true_peaks:
        image += generate_erf_peak(y_coords, x_coords, r, c, sig, amp)

    image = np.random.poisson(image).astype(np.float32)
    image_batch = image[np.newaxis, ...]

    finder = MatrixFreeSparseRBFPeakFinder(
        alpha=None,
        gamma=0.5,
        min_sigma=0.5,
        max_sigma=5.0,
        loss="poisson",
        show_steps=False,
    )

    results = finder.find_peaks_batch(image_batch)
    peaks = results[0]

    assert len(peaks) >= 3

    print("\n--- GEOMETRY DIAGNOSTICS ---")
    for true_r, true_c, true_sig, true_amp in true_peaks:
        dists = np.sqrt((peaks[:, 1] - true_r) ** 2 + (peaks[:, 2] - true_c) ** 2)
        closest_idx = np.argmin(dists)

        p_c, p_r, p_col, p_sig = peaks[closest_idx]
        min_dist = dists[closest_idx]

        print(f"Target: r={true_r:5.1f}, c={true_c:5.1f}, sig={true_sig:4.2f}")
        print(
            f"Found : r={p_r:5.2f}, c={p_col:5.2f}, sig={p_sig:4.2f} | Dist: {min_dist:.3f}"
        )

        # Assert subpixel spatial accuracy
        assert min_dist < 0.75, f"Peak wandered! Dist: {min_dist:.2f}"

        # Assert shape preservation (allowing for dyadic grid snapping)
        assert abs(p_sig - true_sig) < 0.5, (
            f"Sigma collapsed/exploded! True: {true_sig}, Found: {p_sig}"
        )
    print("----------------------------\n")

    medians = np.median(image_batch, axis=(1, 2), keepdims=True)
    bg_map = getattr(finder, "_last_bg_map", medians)

    print("\n--- GHOST PEAK DIAGNOSTICS ---")
    print(f"Total peaks returned by Finder: {len(peaks)} (Expected: 3)")

    # Sort peaks by amplitude (descending)
    peaks_sorted = peaks[np.argsort(peaks[:, 0])[::-1]]

    print("\nTop 10 Peaks by Amplitude:")
    for i, p in enumerate(peaks_sorted[:10]):
        print(
            f"  [{i}] Amp: {p[0]:6.1f} | r: {p[1]:5.1f}, c: {p[2]:5.1f} | sig: {p[3]:4.2f}"
        )

    # Isolate ONLY the True Peaks for the strict deviance check
    # (Filtering out the 0.5-sigma background ghost bumps)
    top_peaks = peaks_sorted[:3]

    metrics_top = finder.compute_metrics(
        image_batch, bg_map, [top_peaks], global_max=1.0
    )
    deviance_top = metrics_top["deviance_nu"]
    print(f"\nDeviance (Top 3 True Peaks Only): {deviance_top:.3f}")

    metrics = finder.compute_metrics(image_batch, bg_map, [peaks], global_max=1.0)
    deviance = metrics["deviance_nu"]

    assert deviance < 1.5


def test_large_sensor_basic_recovery_finder():
    """
    Diagnostic Test: Simulates a 512x512 detector with a FLAT Poisson background.
    This isolates whether the GPU batching, memory, and scaling work on large arrays
    without the confounding variable of morphological halo errors.
    """
    import numpy as np
    import scipy.special

    from subhkl.search.matrix_free import MatrixFreeSparseRBFPeakFinder

    H, W = 512, 512
    np.random.seed(42)

    y_coords, x_coords = np.meshgrid(np.arange(H), np.arange(W), indexing="ij")

    # Flat background: 20 photons.
    image = np.random.poisson(20.0, size=(H, W)).astype(np.float32)

    # Inject 3 strong, isolated peaks
    true_peaks = [
        (100.0, 100.0, 2.0, 300.0),
        (400.0, 400.0, 2.0, 250.0),
        (150.0, 350.0, 1.5, 400.0),
    ]
    for r, c, sig, amp in true_peaks:
        sig_sq2 = sig * np.sqrt(2.0) + 1e-6
        erf_y = scipy.special.erf((y_coords + 0.5 - r) / sig_sq2) - scipy.special.erf(
            (y_coords - 0.5 - r) / sig_sq2
        )
        erf_x = scipy.special.erf((x_coords + 0.5 - c) / sig_sq2) - scipy.special.erf(
            (x_coords - 0.5 - c) / sig_sq2
        )
        image += amp * (np.pi / 2.0) * (sig**2) * erf_y * erf_x

    image_batch = image[np.newaxis, ...]

    finder = MatrixFreeSparseRBFPeakFinder(
        alpha=None,
        gamma=0.5,
        min_sigma=1.0,
        max_sigma=5.0,
        loss="poisson",
        show_steps=False,
    )

    results = finder.find_peaks_batch(image_batch)
    peaks = results[0]

    # 1. Did it explode into noise?
    assert len(peaks) < 15, (
        f"Basic Finder Failed: Hallucinated {len(peaks)} peaks on a flat background!"
    )

    # 2. Did it find the 3 real ones?
    assert len(peaks) >= 3, (
        f"Basic Finder Failed: Missed the real peaks, only found {len(peaks)}."
    )


def test_large_sensor_artifact_suppression():
    """
    Simulates a full 512x512 detector panel with a massive, curved
    diffuse scattering background (halo) to ensure the solver does NOT
    hallucinate a grid of false peaks to fit the unmodeled background curvature.
    """
    import numpy as np
    import scipy.special

    from subhkl.search.matrix_free import MatrixFreeSparseRBFPeakFinder

    H, W = 512, 512
    np.random.seed(42)

    y_coords, x_coords = np.meshgrid(np.arange(H), np.arange(W), indexing="ij")

    # 1. Create a massive, curved halo that a flat plane CANNOT fit
    r2_halo = (x_coords - 256) ** 2 + (y_coords - 256) ** 2
    bg_curved = 20.0 + 150.0 * np.exp(-r2_halo / (2 * 100**2))

    image = np.random.poisson(bg_curved).astype(np.float32)

    # 2. Inject exactly TWO real peaks
    true_peaks = [(100.0, 100.0, 1.5, 200.0), (400.0, 400.0, 2.0, 250.0)]
    for r, c, sig, amp in true_peaks:
        sig_sq2 = sig * np.sqrt(2.0) + 1e-6
        erf_y = scipy.special.erf((y_coords + 0.5 - r) / sig_sq2) - scipy.special.erf(
            (y_coords - 0.5 - r) / sig_sq2
        )
        erf_x = scipy.special.erf((x_coords + 0.5 - c) / sig_sq2) - scipy.special.erf(
            (x_coords - 0.5 - c) / sig_sq2
        )
        phi = (np.pi / 2.0) * (sig**2) * erf_y * erf_x
        image += amp * phi

    image_batch = image[np.newaxis, ...]

    # Test Peak Finder robustness to background curvature
    finder = MatrixFreeSparseRBFPeakFinder(
        alpha=None,
        gamma=0.5,
        min_sigma=1.0,
        max_sigma=5.0,
        loss="poisson",
        show_steps=False,
    )

    results = finder.find_peaks_batch(image_batch)
    peaks = results[0]

    # The solver MUST NOT hallucinate grids.
    # We allow a small buffer for extreme Poisson noise spikes, but it absolutely cannot be > 10.
    assert len(peaks) >= 2, f"Failed to find the 2 real peaks, found {len(peaks)}"
    assert len(peaks) < 10, (
        f"Grid Pathology! Hallucinated {len(peaks)} peaks to fit the background."
    )


def test_large_sensor_basic_integration():
    """
    Diagnostic Test: Tests the dense SSN Integrator on a 512x512 array with a
    FLAT Poisson background. This proves whether the massive Ht @ u matrix
    operations and active-set thresholding are intact for large arrays.
    """
    try:
        from subhkl.peakfinder.sparse_rbf import integrate_peaks_rbf_ssn
    except ImportError:
        from subhkl.search.sparse_rbf import integrate_peaks_rbf_ssn

    import numpy as np
    import scipy.special

    H, W = 512, 512
    np.random.seed(101)

    y_coords, x_coords = np.meshgrid(np.arange(H), np.arange(W), indexing="ij")

    # Flat background
    image = np.random.poisson(15.0, size=(H, W)).astype(np.float32)

    true_r, true_c = 256.0, 256.0
    true_sig, true_amp = 2.0, 200.0
    sig_sq2 = true_sig * np.sqrt(2.0) + 1e-6
    erf_y = scipy.special.erf((y_coords + 0.5 - true_r) / sig_sq2) - scipy.special.erf(
        (y_coords - 0.5 - true_r) / sig_sq2
    )
    erf_x = scipy.special.erf((x_coords + 0.5 - true_c) / sig_sq2) - scipy.special.erf(
        (x_coords - 0.5 - true_c) / sig_sq2
    )
    image += true_amp * (np.pi / 2.0) * (true_sig**2) * erf_y * erf_x

    # Mock orchestrator
    class MockImageHandler:
        def __init__(self, ims):
            self.ims = ims
            self.bank_mapping = {0: 1}

        def get_run_id(self, img_key):
            return 0

    class MockPeaks:
        def __init__(self, ims):
            self.image = MockImageHandler(ims)

        def get_run_id(self, img_key):
            return self.image.get_run_id(img_key)

        def get_detector_by_img(self, img_key):
            from subhkl.instrument.detector import Detector

            return Detector(
                {
                    "n": H,
                    "m": W,
                    "width": W,
                    "height": H,
                    "pixel_size": 1.0,
                    "center": [0, 0, 100],
                    "uhat": [1, 0, 0],
                    "vhat": [0, 1, 0],
                    "panel": "flat",
                }
            )

    # Predictions: 1 True peak (Index 0), 2 Fake peaks far away
    pred_r = np.array([true_r, 50.0, 450.0])
    pred_c = np.array([true_c, 50.0, 450.0])

    # Use distinct, non-harmonic Miller indices so the deduplicator keeps them all
    h_arr = np.array([1, 2, 3])
    k_arr = np.array([13, 17, 19])
    l_arr = np.array([1, 1, 1])

    peak_dict = {0: [pred_r, pred_c, h_arr, k_arr, l_arr, np.ones(3)]}

    res = integrate_peaks_rbf_ssn(
        peak_dict=peak_dict,
        peaks_obj=MockPeaks({0: image}),
        sigmas=[1.0, 2.0, 4.0],
        alpha=4.0,
        # gamma=1 is safe here, unlike in the finder: integration is given the
        # peak positions and does no model selection over them, so the scale
        # degeneracy of docs/matrix_free_theory.md Theorem 1 does not apply.
        gamma=1.0,
        show_progress=False,
    )

    intensities = np.array(res.intensity)
    sigIs = np.array(res.sigma)  # We are now properly returning statistical uncertainty
    snrs = intensities / (sigIs + 1e-9)

    # 1. Did it differentiate real vs fake?
    assert snrs[0] > 3.0, f"True peak SNR too low! Expected > 3.0, got {snrs[0]:.2f}"

    # Empty background measurements will fluctuate due to OLS on Poisson noise.
    # We assert that the solver mathematically recognizes them as insignificant (SNR < 3.0)
    assert snrs[1] < 3.0, (
        f"Fake peak 1 hallucinated mass! SNR: {snrs[1]:.2f}, Mass: {intensities[1]:.2f}"
    )
    assert snrs[2] < 3.0, (
        f"Fake peak 2 hallucinated mass! SNR: {snrs[2]:.2f}, Mass: {intensities[2]:.2f}"
    )

    # 2. Did the unpenalized Measurement Phase (NNLS) correctly measure the unbiased mass?
    expected_intensity = true_amp * 2 * np.pi * true_sig**2
    found_intensity = intensities[0]

    assert np.isclose(found_intensity, expected_intensity, rtol=0.15), (
        f"Debiasing failed: {found_intensity} vs {expected_intensity}"
    )


def test_integrator_large_sensor_halo_suppression():
    """
    Validates that the dense matrix GPU integrator successfully subtracts the
    complex morphological halo before evaluation, and properly executes the
    debiasing loop to prevent real peaks from being crushed by the L1 penalty.
    """
    try:
        from subhkl.peakfinder.sparse_rbf import integrate_peaks_rbf_ssn
    except ImportError:
        from subhkl.search.sparse_rbf import integrate_peaks_rbf_ssn

    import numpy as np
    import scipy.special

    H, W = 512, 512
    np.random.seed(101)

    y_coords, x_coords = np.meshgrid(np.arange(H), np.arange(W), indexing="ij")
    r2_halo = (x_coords - 256) ** 2 + (y_coords - 256) ** 2
    bg_curved = 15.0 + 100.0 * np.exp(-r2_halo / (2 * 120**2))

    image = np.random.poisson(bg_curved).astype(np.float32)

    true_r, true_c = 300.0, 300.0
    true_sig, true_amp = 2.0, 150.0
    sig_sq2 = true_sig * np.sqrt(2.0) + 1e-6
    erf_y = scipy.special.erf((y_coords + 0.5 - true_r) / sig_sq2) - scipy.special.erf(
        (y_coords - 0.5 - true_r) / sig_sq2
    )
    erf_x = scipy.special.erf((x_coords + 0.5 - true_c) / sig_sq2) - scipy.special.erf(
        (x_coords - 0.5 - true_c) / sig_sq2
    )
    image += true_amp * (np.pi / 2.0) * (true_sig**2) * erf_y * erf_x

    # Mocking framework for the Orchestrator
    class MockImageHandler:
        def __init__(self, ims):
            self.ims = ims
            self.bank_mapping = {0: 1}

        def get_run_id(self, img_key):
            return 0

    class MockPeaks:
        def __init__(self, ims):
            self.image = MockImageHandler(ims)

        def get_run_id(self, img_key):
            return self.image.get_run_id(img_key)

        def get_detector_by_img(self, img_key):
            from subhkl.instrument.detector import Detector

            return Detector(
                {
                    "n": H,
                    "m": W,
                    "width": W,
                    "height": H,
                    "pixel_size": 1.0,
                    "center": [0, 0, 100],
                    "uhat": [1, 0, 0],
                    "vhat": [0, 1, 0],
                    "panel": "flat",
                }
            )

    # Provide a grid of HKL predictions. Only ONE matches the true peak (Index 5).
    grid_i, grid_j = np.linspace(50, 450, 10), np.linspace(50, 450, 10)
    grid_i[5], grid_j[5] = true_r, true_c

    # Generate 10 unique fundamental rays
    h_arr = np.arange(1, 11)
    k_arr = np.full(10, 13)
    l_arr = np.full(10, 17)

    peak_dict = {0: [grid_i, grid_j, h_arr, k_arr, l_arr, np.ones(10)]}

    res = integrate_peaks_rbf_ssn(
        peak_dict=peak_dict,
        peaks_obj=MockPeaks({0: image}),
        sigmas=[1.0, 2.0, 4.0],
        alpha=5.0,
        # gamma=1 is safe here, unlike in the finder: integration is given the
        # peak positions and does no model selection over them, so the scale
        # degeneracy of docs/matrix_free_theory.md Theorem 1 does not apply.
        gamma=1.0,
        show_progress=False,
    )

    intensities = np.array(res.intensity)
    sigIs = np.array(res.sigma)
    snrs = intensities / (sigIs + 1e-9)

    # 1. The integrator measures everything, so we rely on SNR to reject halos
    assert snrs[5] > 3.0, f"True peak SNR too low! Expected > 3.0, got {snrs[5]:.2f}"

    # Check that all other 9 fake halo points are rejected by high uncertainty
    fake_indices = [i for i in range(10) if i != 5]
    for i in fake_indices:
        # A successful "Halo Trap" means the target's unconstrained intensity is statistically
        # insignificant compared to the local background variance.
        assert snrs[i] < 3.0, (
            f"Halo trap failed! Fake peak {i} has high SNR: {snrs[i]:.2f} (Mass: {intensities[i]:.2f})"
        )

    # 2. The debiasing loop must recover the full intensity
    expected_intensity = true_amp * 2 * np.pi * true_sig**2
    found_intensity = intensities[5]

    # Allow 15% tolerance for Poisson noise variance
    assert np.isclose(found_intensity, expected_intensity, rtol=0.15), (
        f"Halo Debias failed: {found_intensity} vs {expected_intensity}"
    )


def test_poisson_local_variance_suppression():
    """
    Regression test for exact Poisson local variance.
    Injects two identical weak peaks: one on a dark background (low variance)
    and one on a bright halo (high variance).
    The spatially varying 1/U_k variance map MUST suppress the peak on the bright halo
    while preserving the peak on the dark background.
    """
    import numpy as np
    import scipy.special

    from subhkl.search.matrix_free import MatrixFreeSparseRBFPeakFinder

    H, W = 100, 100
    np.random.seed(42)

    y_coords, x_coords = np.meshgrid(np.arange(H), np.arange(W), indexing="ij")

    # 1. Base flat background (Dark / Low Noise) -> expected variance ~ 10
    bg_flat = 10.0
    image = np.full((H, W), bg_flat, dtype=np.float32)

    # 2. Add a massive, bright diffuse structure (Bright / High Noise) -> expected variance ~ 510
    halo_r, halo_c = 50.0, 75.0
    r2_halo = (x_coords - halo_c) ** 2 + (y_coords - halo_r) ** 2
    image += 500.0 * np.exp(-r2_halo / (2 * 15**2))

    def generate_erf_peak(y, x, r, c, sig, amp):
        sig_sq2 = sig * np.sqrt(2.0) + 1e-6
        erf_y = scipy.special.erf((y + 0.5 - r) / sig_sq2) - scipy.special.erf(
            (y - 0.5 - r) / sig_sq2
        )
        erf_x = scipy.special.erf((x + 0.5 - c) / sig_sq2) - scipy.special.erf(
            (x - 0.5 - c) / sig_sq2
        )
        return amp * (np.pi / 2.0) * (sig**2) * erf_y * erf_x

    # 3. Inject two IDENTICAL weak peaks
    peak_a_r, peak_a_c = 50.0, 25.0  # Peak A: On the dark background
    peak_b_r, peak_b_c = 50.0, 75.0  # Peak B: Dead center on the bright halo

    # A matched sigma=1.5 atom integrates ~pi*sigma^2 pixels of evidence, so
    # its significance is z = amp * sqrt(pi * sig**2 / u), not amp / sqrt(u):
    # amp=20 gives z ~ 17 on the dark background (u = 10) and z ~ 2.4 on the
    # bright region (u = 510).  Against the alpha=None false-alarm floor
    # (~4.1 at sigma=1 for this frame), A clears by 4x and B sits 1.7 sigma
    # below.  The previous amp=60 put B at z ~ 7.1: above the floor, so
    # suppressing it needed the hand-picked alpha=8, and even then by only
    # 0.9 sigma -- an accidental margin, not a designed one.
    test_amp = 20.0
    test_sig = 1.5

    image += generate_erf_peak(
        y_coords, x_coords, peak_a_r, peak_a_c, test_sig, test_amp
    )
    image += generate_erf_peak(
        y_coords, x_coords, peak_b_r, peak_b_c, test_sig, test_amp
    )

    # Apply true Poisson noise
    image = np.random.poisson(image).astype(np.float32)
    image_batch = image[np.newaxis, ...]

    # 4. Configure Finder: alpha=None puts the threshold at the false-alarm
    # floor; see the test_amp comment for the matched-filter margins.
    finder = MatrixFreeSparseRBFPeakFinder(
        alpha=None,
        gamma=0.5,
        min_sigma=1.0,
        max_sigma=5.0,
        loss="poisson",
        show_steps=False,
    )

    results = finder.find_peaks_batch(image_batch)
    peaks = results[0]

    found_a = False
    found_b = False

    # Count an atom as A or B only if its width is commensurate with the
    # injected sigma = 1.5 (the sigma bank is {1..5}; a genuine detection
    # lands on 1 or 2).  The morphological median background under-fits the
    # bright region's interior, and the solver absorbs that residual with
    # atoms -- narrow ones fencing its edges and broad ones (sigma ~ 4) near
    # its centre.  Those are background artifacts, not peak detections, and
    # where one lands relative to B is a coin toss that must not decide this
    # assertion.  The previous gate (sigma < 0.98 * max_sigma) only excluded
    # atoms pinned at the bank edge, which stopped working the moment the
    # solver converged well enough to fit that residual at sigma = 4.
    def _is_peak_like(p):
        return p[3] <= 2.5

    for p in peaks:
        # p = [intensity, r, c, sigma]
        if not _is_peak_like(p):
            continue
        if np.sqrt((p[1] - peak_a_r) ** 2 + (p[2] - peak_a_c) ** 2) < 2.0:
            found_a = True
        if np.sqrt((p[1] - peak_b_r) ** 2 + (p[2] - peak_b_c) ** 2) < 2.0:
            found_b = True

    assert found_a, "Failed to find the weak peak in the low-variance (dark) region."
    assert not found_b, (
        "Incorrectly found the weak peak in the high-variance (bright) region! The local variance map did not suppress it."
    )


def test_poisson_subpatch_variance_suppression():
    """
    Regression test explicitly isolating pixel-level 1/U_k variance.
    A bright 24x24 plateau is injected. It is large enough to survive
    the 25x25 morphological median filter, but small enough that the
    median of the 94x94 evaluation patch remains the dark background (10.0).

    The old Gaussian model would assign a global patch noise floor of sqrt(10)
    and falsely detect Peak B. The new Poisson model evaluates local variance
    as sqrt(510) and correctly suppresses it.
    """
    import numpy as np
    import scipy.special

    from subhkl.search.matrix_free import MatrixFreeSparseRBFPeakFinder

    H, W = 128, 128
    np.random.seed(42)

    y_coords, x_coords = np.meshgrid(np.arange(H), np.arange(W), indexing="ij")

    # 1. Base flat background (Dark / Low Noise) -> expected variance ~ 10
    bg_flat = 10.0
    image = np.full((H, W), bg_flat, dtype=np.float32)

    # 2. Sub-Patch Plateau (Bright / High Noise) -> expected variance ~ 510
    # Must be > 18x18 to survive a 25x25 median filter.
    # Must be < 30x30 so it doesn't inflate the median of the 94x94 patch.
    plateau_min, plateau_max = 52, 76  # 24x24 square
    image[plateau_min:plateau_max, plateau_min:plateau_max] += 500.0

    def generate_erf_peak(y, x, r, c, sig, amp):
        sig_sq2 = sig * np.sqrt(2.0) + 1e-6
        erf_y = scipy.special.erf((y + 0.5 - r) / sig_sq2) - scipy.special.erf(
            (y - 0.5 - r) / sig_sq2
        )
        erf_x = scipy.special.erf((x + 0.5 - c) / sig_sq2) - scipy.special.erf(
            (x - 0.5 - c) / sig_sq2
        )
        return amp * (np.pi / 2.0) * (sig**2) * erf_y * erf_x

    # 3. Inject two IDENTICAL weak peaks
    peak_a_r, peak_a_c = 25.0, 25.0  # Peak A: On the dark background
    peak_b_r, peak_b_c = 64.0, 64.0  # Peak B: Dead center on the bright plateau

    # A matched sigma=1.5 atom integrates ~pi*sigma^2 pixels of evidence, so
    # its significance is z = amp * sqrt(pi * sig**2 / u), not amp / sqrt(u):
    # amp=20 gives z ~ 17 on the dark background (u = 10) and z ~ 2.4 on the
    # bright region (u = 510).  Against the alpha=None false-alarm floor
    # (~4.1 at sigma=1 for this frame), A clears by 4x and B sits 1.7 sigma
    # below.  The previous amp=60 put B at z ~ 7.1: above the floor, so
    # suppressing it needed the hand-picked alpha=8, and even then by only
    # 0.9 sigma -- an accidental margin, not a designed one.
    test_amp = 20.0
    test_sig = 1.5

    image += generate_erf_peak(
        y_coords, x_coords, peak_a_r, peak_a_c, test_sig, test_amp
    )
    image += generate_erf_peak(
        y_coords, x_coords, peak_b_r, peak_b_c, test_sig, test_amp
    )

    image = np.random.poisson(image).astype(np.float32)
    image_batch = image[np.newaxis, ...]

    # 4. Configure Finder: alpha=None puts the threshold at the false-alarm
    # floor; see the test_amp comment for the matched-filter margins.  (A
    # Gaussian model using the global patch median=10 would score B at z ~ 17
    # and falsely detect it; the Poisson 1/U map scores it at z ~ 2.4.)
    finder = MatrixFreeSparseRBFPeakFinder(
        alpha=None,
        gamma=0.5,
        min_sigma=1.0,
        max_sigma=5.0,
        loss="poisson",
        show_steps=False,
    )

    results = finder.find_peaks_batch(image_batch)
    peaks = results[0]

    found_a = False
    found_b = False

    # Count an atom as A or B only if its width is commensurate with the
    # injected sigma = 1.5 (the sigma bank is {1..5}; a genuine detection
    # lands on 1 or 2).  The morphological median background under-fits the
    # bright region's interior, and the solver absorbs that residual with
    # atoms -- narrow ones fencing its edges and broad ones (sigma ~ 4) near
    # its centre.  Those are background artifacts, not peak detections, and
    # where one lands relative to B is a coin toss that must not decide this
    # assertion.  The previous gate (sigma < 0.98 * max_sigma) only excluded
    # atoms pinned at the bank edge, which stopped working the moment the
    # solver converged well enough to fit that residual at sigma = 4.
    def _is_peak_like(p):
        return p[3] <= 2.5

    for p in peaks:
        # p = [intensity, r, c, sigma]
        if not _is_peak_like(p):
            continue
        if np.sqrt((p[1] - peak_a_r) ** 2 + (p[2] - peak_a_c) ** 2) < 2.0:
            found_a = True
        if np.sqrt((p[1] - peak_b_r) ** 2 + (p[2] - peak_b_c) ** 2) < 2.0:
            found_b = True

    assert found_a, "Failed to find the weak peak in the low-variance (dark) region."
    assert not found_b, (
        "Regression Failed: Incorrectly found the weak peak on the intense plateau. The exact 1/U_k map did not apply!"
    )


def test_boundary_sigma_rejection_fires_on_unmodelled_background():
    """The sigma-at-bank-edge filter must actually trigger, and only on background.

    A broad smooth halo is under-fitted by the morphological background estimator
    at its centre, and the residual is picked up as an atom whose width runs to
    ``max_sigma`` -- the solver asking for a wider basis than it was given.  That
    is the signature of unmodelled background rather than of a reflection, since
    a real peak's width is set by the point-spread function and lands inside the
    bank.

    This asserts the filter both fires here and does not eat the genuine peak
    placed well away from the halo.  See docs/matrix_free_theory.md section 7b.
    """
    import numpy as np

    from subhkl.search.matrix_free import MatrixFreeSparseRBFPeakFinder

    H, W = 100, 100
    np.random.seed(7)
    y_coords, x_coords = np.meshgrid(np.arange(H), np.arange(W), indexing="ij")

    image = np.full((H, W), 10.0, dtype=np.float32)
    # Broad diffuse halo: far wider than max_sigma, so no single basis function
    # can represent it and the background estimator under-fits its centre.
    image += 500.0 * np.exp(
        -((x_coords - 75.0) ** 2 + (y_coords - 50.0) ** 2) / (2 * 15.0**2)
    )
    # A genuine, well-resolved peak far from the halo.
    image += generate_erf_peak(y_coords, x_coords, 50.0, 20.0, 1.5, 200.0)
    image = np.random.poisson(image).astype(np.float32)
    image_batch = image[np.newaxis, ...]

    max_sigma = 5.0
    kwargs = {
        "alpha": 8.0,
        "gamma": 0.5,
        "min_sigma": 1.0,
        "max_sigma": max_sigma,
        "loss": "poisson",
        "show_steps": False,
    }

    # With the filter off, the halo residual is reported, pinned at the bank edge.
    unfiltered = MatrixFreeSparseRBFPeakFinder(reject_boundary_sigma=False, **kwargs)
    raw = unfiltered.find_peaks_batch(image_batch)[0]
    pinned = [p for p in raw if p[3] >= 0.98 * max_sigma]
    assert len(pinned) >= 1, (
        "expected the unmodelled halo to produce at least one atom pinned at "
        f"max_sigma, got widths {sorted(round(float(p[3]), 2) for p in raw)}"
    )

    # With the filter on, those atoms are gone and the count is reported.
    filtered = MatrixFreeSparseRBFPeakFinder(reject_boundary_sigma=True, **kwargs)
    kept = filtered.find_peaks_batch(image_batch)[0]

    assert filtered.n_boundary_rejected[0] >= 1, (
        "the boundary-sigma filter did not fire on a case built to trigger it"
    )
    assert all(p[3] < 0.98 * max_sigma for p in kept), (
        "an atom pinned at max_sigma survived the filter"
    )

    # The real peak must survive: the filter must reject background, not signal.
    assert any(np.sqrt((p[1] - 50.0) ** 2 + (p[2] - 20.0) ** 2) < 2.0 for p in kept), (
        "the boundary-sigma filter removed the genuine peak"
    )


def test_alpha_none_derives_threshold_from_the_false_alarm_floor():
    """`alpha=None` should set the threshold level from the data, not a constant.

    Solving globally tests every (pixel, scale) coefficient at once, so the
    level that keeps the expected number of false detections at O(1) over the
    image is ``sqrt(2 log N_k)`` with ``N_k`` the resolution-element count at
    that scale.  That level depends on how big the image is, which a hard-coded
    constant cannot express: the same constant is too strict for a small crop
    and too permissive for a full detector frame.

    What must be preserved is the ``sigma**gamma`` *shape* of the threshold,
    since that is what sets the merge/split balance.  Using the floor alone
    flattens it and over-merges, losing weak peaks in the tails of strong ones.
    """
    import numpy as np

    from subhkl.search.matrix_free import MatrixFreeSparseRBFPeakFinder

    gamma = 0.5
    finder = MatrixFreeSparseRBFPeakFinder(
        alpha=None, gamma=gamma, min_sigma=1.0, max_sigma=5.0
    )
    sigmas = np.array(finder.sigmas)
    weights = (sigmas / finder.ref_sigma) ** gamma

    def floor_for(side):
        n = np.maximum((side * side) / (2 * np.pi * sigmas**2), 2.0)
        return np.sqrt(2 * np.log(n))

    a_small = np.array(finder.effective_alpha(64, 64))
    a_large = np.array(finder.effective_alpha(4096, 4096))

    # It must never sit below the false-alarm floor at any scale.
    assert np.all(a_small >= floor_for(64) - 1e-4)
    assert np.all(a_large >= floor_for(4096) - 1e-4)

    # It must be the *smallest* such level, i.e. touching the floor somewhere,
    # otherwise it is needlessly throwing away real peaks.
    assert np.isclose(np.min(a_small - floor_for(64)), 0.0, atol=1e-4)

    # The sigma**gamma shape must survive: alpha_eff / w is one constant.
    assert np.allclose(a_small / weights, (a_small / weights)[0])

    # A bigger image tests more coefficients, so it must demand more evidence.
    assert np.all(a_large > a_small)

    # An explicit alpha is a lower bound on significance, not a way under the
    # floor: too small a request is raised, a strict one is honoured.
    lax_finder = MatrixFreeSparseRBFPeakFinder(
        alpha=0.01, gamma=gamma, min_sigma=1.0, max_sigma=5.0
    )
    assert np.allclose(np.array(lax_finder.effective_alpha(130, 130)), floor_for(130))

    strict = MatrixFreeSparseRBFPeakFinder(
        alpha=20.0, gamma=gamma, min_sigma=1.0, max_sigma=5.0
    )
    assert np.all(np.array(strict.effective_alpha(130, 130)) > floor_for(130))

    # And it must still work: a clear peak on a flat background is found.
    H = W = 60
    np.random.seed(11)
    y_coords, x_coords = np.meshgrid(np.arange(H), np.arange(W), indexing="ij")
    image = np.full((H, W), 20.0, dtype=np.float32)
    image += generate_erf_peak(y_coords, x_coords, 30.0, 30.0, 2.0, 300.0)
    image = np.random.poisson(image).astype(np.float32)

    peaks = finder.find_peaks_batch(image[np.newaxis, ...])[0]
    assert len(peaks) >= 1
    assert any(np.sqrt((p[1] - 30.0) ** 2 + (p[2] - 30.0) ** 2) < 2.0 for p in peaks), (
        "alpha=None failed to find an unambiguous peak"
    )


def test_global_solve_reaches_first_order_optimality():
    """The global solve must return a certified first-order optimum.

    The optimality measure is the per-coordinate residual of the
    prox-gradient fixed-point map, relative to the local penalty scale:

        r = max_i |c_i - prox_i| / (tau * lam_i),
        prox = max(0, c - tau * (grad D + lam)),

    which is zero exactly at a KKT point and dimensionless (a residual of
    r = 1 means the first-order error is as large as the penalty that
    decides activation).  The solver's internal stopping test certifies
    max|G|/lam <= 1e-3 in the q-iterate, and the c-space measure asserted
    here is dominated by it (the soft-threshold is 1-Lipschitz).

    This protects two coupled solver properties against regression:
    - sufficient-decrease acceptance: the Newton step is taken only when
      it decreases J at least as much as the certified forward-backward
      step, so every iteration is provably at least as good as FB;
    - the relative KKT stopping test: the previous absolute step-norm
      test (||dq|| <= 1e-3) fired at a residual of the ORDER OF the
      penalty itself (r ~ 2-7 on this case) once small FB steps appeared,
      reading "steps got small" as "converged".

    Measured on this case: r = 4.4e-4 at the stop (float32 plateau
    ~2e-5 with unlimited iterations), versus r ~ 91 at c = 0.  The 2e-3
    threshold is 2x the internal certificate for float32 slack and still
    three orders below the pre-fix behaviour.
    """
    import jax.numpy as jnp
    import numpy as np
    from jax import lax

    from subhkl.search.matrix_free import MatrixFreeSparseRBFPeakFinder
    from subhkl.search.sparse_rbf import compute_bg_batch

    rng = np.random.default_rng(0)
    H, W = 96, 96
    img = np.full((H, W), 5.0)
    for _ in range(4):
        r0, c0 = rng.integers(20, H - 20), rng.integers(20, W - 20)
        yy, xx = np.ogrid[:H, :W]
        img += 300.0 * np.exp(-((yy - r0) ** 2 + (xx - c0) ** 2) / (2 * 2.0**2))
    image = rng.poisson(img).astype(np.float32)

    finder = MatrixFreeSparseRBFPeakFinder(
        alpha=None, gamma=0.5, max_sigma=5.0, num_sigmas=5, loss="poisson"
    )
    filter_size = max(15, int(finder.max_sigma * 5))
    bg = np.asarray(compute_bg_batch(jnp.asarray(image[None]), filter_size))[0]

    pad = (2 * finder.max_k_rad + 1) // 2
    ip = jnp.pad(jnp.asarray(image), ((pad, pad), (pad, pad)), mode="edge")
    bp = jnp.pad(jnp.asarray(bg), ((pad, pad), (pad, pad)), mode="edge")

    c_sol = finder._solve_ssn_cg_global(ip, bp, max_iter=400)[None]

    # The measure is built from first principles with the test's own step
    # size: c is a KKT point iff c = prox(c - tau grad) for ANY tau > 0,
    # so an independently derived tau cannot mask a solver regression.
    y4 = ip[None, None, :, :]
    bg4 = bp[None, None, :, :]
    w_ref = 1.0 / jnp.maximum(bg4, 1e-3)
    h_diag = jnp.maximum(finder._adjoint_op(w_ref, finder.K_sq), 1e-6)
    lam = (
        finder.effective_alpha(*ip.shape)[None, :, None, None]
        * h_diag
        * jnp.sqrt(1.0 / h_diag)
    )

    def power_step(_, v):
        av = finder._adjoint_op(
            w_ref * finder._forward_op(v, finder.K_weights), finder.K_weights
        )
        return av / (jnp.linalg.norm(av) + 1e-12)

    v0 = jnp.ones_like(lam)
    v_top = lax.fori_loop(0, 15, power_step, v0 / jnp.linalg.norm(v0))
    av_top = finder._adjoint_op(
        w_ref * finder._forward_op(v_top, finder.K_weights), finder.K_weights
    )
    l_max = jnp.sum(v_top * av_top) / jnp.sum(v_top * v_top)
    tau = 1.0 / (l_max + 1e-4)

    u = jnp.maximum(finder._forward_op(c_sol, finder.K_weights) + bg4, 1e-6)
    grad = finder._adjoint_op(1.0 - y4 / u, finder.K_weights)
    prox = jnp.maximum(0.0, c_sol - tau * (grad + lam))
    kkt_rel = float(jnp.max(jnp.abs(c_sol - prox) / (tau * lam)))

    assert np.isfinite(kkt_rel)
    assert kkt_rel < 2e-3, (
        f"first-order optimality regressed: max|c - prox|/(tau*lam) = "
        f"{kkt_rel:.3e} (certified <= 1e-3, measured 4.4e-4 at the fix)"
    )
    # sanity: the certified solution is a sparse peak model, not c = 0
    assert int(jnp.sum(c_sol > 0)) > 0


def test_sparse_regime_certificate_and_shadow_deactivation():
    """First-order resolution of the data-shadow degeneracy.

    At background 0.3 counts/pixel, ~74% of pixels record zero counts, so
    the exact Hessian's null space -- directions whose image lands on
    empty pixels -- is most of coefficient space.  The augmented-penalty
    endgame resolves those directions by deactivation (the zero-count
    fidelity terms are exactly linear and fold into the threshold), with
    no Hessian regularization.  Protected properties: the solve still
    reaches its relative-KKT certificate despite the singular Hessian;
    the dark far corner carries exactly zero coefficients (deactivation,
    not damping); and the one real peak is recovered.
    """
    import jax.numpy as jnp
    import numpy as np
    from jax import lax

    from subhkl.search.matrix_free import MatrixFreeSparseRBFPeakFinder

    rng = np.random.default_rng(3)
    H, W = 64, 64
    bg_level = 0.3
    yy, xx = np.meshgrid(np.arange(H), np.arange(W), indexing="ij")
    truth = np.full((H, W), bg_level)
    truth += 3.0 * np.exp(-((yy - 32) ** 2 + (xx - 32) ** 2) / (2 * 2.0**2))
    image = rng.poisson(truth).astype(np.float32)
    assert np.mean(image == 0) > 0.5  # the shadow is most of the frame

    finder = MatrixFreeSparseRBFPeakFinder(
        alpha=None, gamma=0.5, max_sigma=5.0, num_sigmas=5, loss="poisson"
    )
    pad = (2 * finder.max_k_rad + 1) // 2
    ip = jnp.pad(jnp.asarray(image), ((pad, pad), (pad, pad)), mode="edge")
    bp = jnp.full_like(ip, bg_level)

    c_sol = finder._solve_ssn_cg_global(ip, bp, max_iter=400)[None]

    # certificate, from first principles with an independent step size
    y4 = ip[None, None, :, :]
    bg4 = bp[None, None, :, :]
    w_ref = 1.0 / jnp.maximum(bg4, 1e-3)
    h_diag = jnp.maximum(finder._adjoint_op(w_ref, finder.K_sq), 1e-6)
    lam = (
        finder.effective_alpha(*ip.shape)[None, :, None, None]
        * h_diag
        * jnp.sqrt(1.0 / h_diag)
    )

    def power_step(_, v):
        av = finder._adjoint_op(
            w_ref * finder._forward_op(v, finder.K_weights), finder.K_weights
        )
        return av / (jnp.linalg.norm(av) + 1e-12)

    v0 = jnp.ones_like(lam)
    v_top = lax.fori_loop(0, 15, power_step, v0 / jnp.linalg.norm(v0))
    av_top = finder._adjoint_op(
        w_ref * finder._forward_op(v_top, finder.K_weights), finder.K_weights
    )
    tau = 1.0 / (jnp.sum(v_top * av_top) / jnp.sum(v_top * v_top) + 1e-4)
    u = jnp.maximum(finder._forward_op(c_sol, finder.K_weights) + bg4, 1e-6)
    grad = finder._adjoint_op(1.0 - y4 / u, finder.K_weights)
    prox = jnp.maximum(0.0, c_sol - tau * (grad + lam))
    kkt_rel = float(jnp.max(jnp.abs(c_sol - prox) / (tau * lam)))
    assert kkt_rel < 2e-3, f"sparse-regime certificate regressed: {kkt_rel:.3e}"

    # deactivation across the shadow: the far corner (>= 6 sigma from the
    # peak) must be exactly zero in every channel
    corner = np.asarray(c_sol[0, :, pad : pad + 12, pad : pad + 12])
    assert np.all(corner == 0.0), "dark-region coefficients not deactivated"

    # the real peak is recovered
    c_tot = np.asarray(jnp.sum(c_sol[0], axis=0))
    r_hat, c_hat = np.unravel_index(np.argmax(c_tot), c_tot.shape)
    assert abs(r_hat - pad - 32) <= 1.5 and abs(c_hat - pad - 32) <= 1.5


def test_position_certificate_flags_background_saddle():
    """Second-order: the certificate must flag the background-induced
    saddle of a statistically significant atom and pass a well-posed one.

    Construction (deterministic, a legal Poisson realization): flat field
    at the background mean plus a ring of extra counts at d = 2*sigma --
    the residual geometry a width-mismatched atom leaves behind.  The
    fitted atom is significant (matched-filter z ~ 6.5 against a floor of
    ~4.6) yet its position Hessian is indefinite: refinement bifurcates.
    """
    import jax
    import jax.numpy as jnp
    import numpy as np

    from subhkl.search.matrix_free import MatrixFreeSparseRBFPeakFinder

    finder = MatrixFreeSparseRBFPeakFinder(
        alpha=None, gamma=0.5, max_sigma=5.0, num_sigmas=5, loss="poisson"
    )
    H = W = 33
    B, sig = 2.0, 2.0
    yy, xx = np.meshgrid(np.arange(H), np.arange(W), indexing="ij")
    dist = np.sqrt((yy - 16.0) ** 2 + (xx - 16.0) ** 2)

    # saddle case: ring counts at d = 2*sigma
    y_ring = np.full((H, W), B, dtype=np.float32)
    y_ring[np.abs(dist - 4.0) < 0.5] += 5.0
    bg = np.full((H, W), B, dtype=np.float32)

    # fit the amplitude at the (symmetric) center by scalar Newton
    prof = np.exp(-(dist**2) / (2 * sig**2)).astype(np.float32)

    def fit_amp(y):
        amp = 1.0
        yj = jnp.asarray(y)

        def nll(a):
            U = jnp.maximum(jnp.asarray(bg) + a * jnp.asarray(prof), 1e-6)
            return jnp.sum(U - yj * jnp.log(U))

        g = jax.grad(nll)
        h = jax.grad(g)
        for _ in range(60):
            amp = float(np.clip(amp - g(amp) / max(float(h(amp)), 1e-9), 1e-4, 1e6))
        return amp

    amp_ring = fit_amp(y_ring)
    assert amp_ring > 1.5  # significant: z = amp*sqrt(pi sig^2/B) > 4.6
    peaks = np.array([[amp_ring, 16.0, 16.0, sig]], dtype=np.float32)
    eig_tot, _ = finder.position_curvature(y_ring, bg, peaks)
    assert eig_tot[0, 0] < 0.0, (
        f"background saddle not flagged: min-eig {eig_tot[0, 0]:.3e}"
    )

    # control: a genuine, well-posed peak at the same background
    y_good = np.round(B + 6.0 * prof).astype(np.float32)
    peaks_good = np.array([[6.0, 16.0, 16.0, sig]], dtype=np.float32)
    eig_tot_g, _ = finder.position_curvature(y_good, bg, peaks_good)
    assert eig_tot_g[0, 0] > 0.0, (
        f"well-posed peak wrongly flagged: min-eig {eig_tot_g[0, 0]:.3e}"
    )


def test_dark_wall_transverse_silence_stiffness():
    """The silence part of the position curvature is the wall mechanism:
    a straight dark boundary at distance D = sigma contributes transverse
    stiffness ~ 2*pi*amp*Q''(1) ~ 1.5*amp (measured 1.42 with the exact
    pixel-integrated kernel), about 24% of the peak's full position
    Fisher information 2*pi*amp -- and only transverse to the wall.
    """
    import numpy as np

    from subhkl.search.matrix_free import MatrixFreeSparseRBFPeakFinder

    finder = MatrixFreeSparseRBFPeakFinder(
        alpha=None, gamma=0.5, max_sigma=5.0, num_sigmas=5, loss="poisson"
    )
    H = W = 61
    sig, amp = 2.0, 1.0
    y = np.full((H, W), 5.0, dtype=np.float32)
    y[:, 32:] = 0.0  # dark half-plane: wall at distance D = 2 px = sigma
    bg = np.full((H, W), 5.0, dtype=np.float32)
    peaks = np.array([[amp, 30.0, 30.0, sig]], dtype=np.float32)

    _, eig_sil = finder.position_curvature(y, bg, peaks)
    tangent, transverse = eig_sil[0, 0], eig_sil[0, 1]
    fisher = 2.0 * np.pi * amp
    assert 0.15 * fisher < transverse < 0.35 * fisher, (
        f"wall stiffness off calibration: {transverse / fisher:.3f} of Fisher"
    )
    assert abs(tangent) < 0.2 * transverse, (
        f"wall stiffness not transverse: tangent {tangent:.3e} "
        f"vs transverse {transverse:.3e}"
    )
