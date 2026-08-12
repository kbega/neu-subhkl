"""Robust per-patch amplitude fit (Huber IRLS).

Diffuse ridges narrower than the rolling-median background window are
classified as signal, stay in the fit residual, and correlate with the
template -- measured on cg4d-t4-lysozyme as whole banks of negative
median intensity (32-46% of intensities <= 0).  A free constant column
absorbs flat pedestals and a plane cannot represent a smoothed step, so
the fix is robustness: down-weight unmodeled-structure pixels at the
Poisson noise scale.
"""

from __future__ import annotations

import numpy as np

from subhkl.search.sparse_rbf import SparseLaueIntegrator


def _integrate(img, r0, c0, robust):
    integ = SparseLaueIntegrator(
        alpha=3.0,
        min_sigma=1.5,
        max_sigma=5.0,
        gamma=2.0,
        nominal_sigma=2.5,
        anisotropic=True,
        robust_patch_fit=robust,
        chunk_size=8,
        show_steps=False,
    )
    res = integ.integrate_reflections(
        img[None, :, :],
        np.array([0]),
        np.array([r0]),
        np.array([c0]),
        var_us=np.array([2.5**2]),
        var_vs=np.array([2.5**2]),
        cov_uvs=np.array([0.0]),
    )
    return float(np.asarray(res[0])[0])


def _ridge_scene(c0, flux=4000.0, sigma=2.5, bg=20.0, seed=4):
    rng = np.random.default_rng(seed)
    H = W = 96
    img = np.full((H, W), bg)
    img[:, 40:52] += 300.0  # diffuse ridge, narrower than the bg median window
    yy, xx = np.mgrid[0:H, 0:W]
    img += (
        flux
        * np.exp(-0.5 * (((yy - 48) ** 2 + (xx - c0) ** 2) / sigma**2))
        / (2 * np.pi * sigma**2)
    )
    return rng.poisson(np.maximum(img, 0)).astype(float), flux


def test_robust_fit_recovers_flux_beside_a_ridge():
    # Averaged over noise realizations: single Poisson draws carry
    # ~150-count flux noise, so per-seed comparisons flap.
    errs_on, errs_off = [], []
    for seed in (4, 5, 6, 7):
        img, flux = _ridge_scene(c0=58.0, seed=seed)
        errs_off.append(abs(_integrate(img, 48.0, 58.0, robust=False) - flux))
        errs_on.append(abs(_integrate(img, 48.0, 58.0, robust=True) - flux))
    assert np.mean(errs_on) / flux < 0.06


def test_robust_fit_improves_even_with_ridge_in_the_core():
    errs_on, errs_off = [], []
    for seed in (4, 5, 6, 7):
        img, flux = _ridge_scene(c0=56.0, seed=seed)
        errs_off.append(abs(_integrate(img, 48.0, 56.0, robust=False) - flux))
        errs_on.append(abs(_integrate(img, 48.0, 56.0, robust=True) - flux))
    # The ridge-in-core bias is ~30%; robustness must close a real part
    # of it on average.
    assert np.mean(errs_on) < 0.85 * np.mean(errs_off)


def test_robust_fit_is_neutral_on_clean_background():
    rng = np.random.default_rng(7)
    H = W = 96
    sigma, flux, bg = 2.5, 4000.0, 20.0
    yy, xx = np.mgrid[0:H, 0:W]
    img = rng.poisson(
        bg
        + flux
        * np.exp(-0.5 * (((yy - 48) ** 2 + (xx - 48) ** 2) / sigma**2))
        / (2 * np.pi * sigma**2)
    ).astype(float)
    i_off = _integrate(img, 48.0, 48.0, robust=False)
    i_on = _integrate(img, 48.0, 48.0, robust=True)
    assert abs(i_on - i_off) / flux < 0.05


def test_core_protection_never_clips_a_bright_non_gaussian_peak():
    """Bright cores deviate from the Gaussian template by many Poisson
    sigma; without model-based core protection the Huber pass clips
    them (measured as a 2.5x drop in healthy runs' median intensity)."""
    rng = np.random.default_rng(3)
    H = W = 96
    flux, bg = 60000.0, 20.0
    yy, xx = np.mgrid[0:H, 0:W]
    r2 = (yy - 48.0) ** 2 + (xx - 48.0) ** 2
    moffat = (1 + r2 / (2.0 * 2.5**2)) ** (-2.5)
    moffat = moffat / moffat.sum() * flux
    img = rng.poisson(bg + moffat).astype(float)
    i_off = _integrate(img, 48.0, 48.0, robust=False)
    i_on = _integrate(img, 48.0, 48.0, robust=True)
    assert abs(i_on - i_off) / i_off < 0.02
