import numpy as np
import numpy.typing as npt
from subhkl.core.crystallography import get_q_lab


def scale_coordinates(xp, yp, scale_x, scale_y, nx, ny):
    """
    Scale pixel coordinates to physical coordinates.

    Parameters
    ----------
    xp : float
        Pixel x-coordinate
    yp : float
        Pixel y-coordinate
    scale_x : float
        Scale factor in x direction (m/pixel)
    scale_y : float
        Scale factor in y direction (m/pixel)
    nx : int
        Number of pixels in x direction
    ny : int
        Number of pixels in y direction

    Returns
    -------
    x, y : tuple of float
        Physical coordinates in meters
    """
    x = (xp - nx / 2) * scale_x
    y = (yp - ny / 2) * scale_y
    return x, y


def predict_reflections_on_panel(
    detector,
    h,
    k,
    l,
    RUB,
    wavelength_min,
    wavelength_max,
    sample_offset=None,
    ki_vec=None,
    R_all=None,
    gonio_axes=None,
    gonio_angles=None,
    gonio_offsets=None,
):
    """
    Predict reflection positions on a specific detector panel.
    Expects RUB to be a SINGLE (3,3) matrix for this specific panel's exposure.
    Expects h, k, l to be 1D arrays of length M (the theoretical reflection pool).
    """
    if ki_vec is None:
        ki_vec = np.array([0.0, 0.0, 1.0])

    ki_hat = ki_vec / np.linalg.norm(ki_vec)
    hkl = np.stack([h, k, l], axis=0)  # (3, M)

    # 1. Transform Miller indices to absolute Lab frame Q-vectors
    q_lab = RUB @ hkl
    Q_vec = 2 * np.pi * q_lab  # (3, M)

    Q_sq = np.sum(Q_vec**2, axis=0)  # (M,)
    Q_dot_ki = np.sum(Q_vec * ki_hat[:, None], axis=0)  # (M,)

    # 2. Solve Laue Condition for Wavelength
    with np.errstate(divide="ignore", invalid="ignore"):
        lamda = -4 * np.pi * Q_dot_ki / Q_sq  # (M,)

    mask_wl = (lamda >= wavelength_min) & (lamda <= wavelength_max) & np.isfinite(lamda)

    if not np.any(mask_wl):
        return (
            np.array([]), np.array([]), np.array([]),
            np.array([]), np.array([]), np.array([])
        )

    lamda_v = lamda[mask_wl]
    Q_vec_v = Q_vec[:, mask_wl]
    h_v, k_v, l_v = h[mask_wl], k[mask_wl], l[mask_wl]


    # 3. Calculate Final Scattered Ray Direction (kf)
    k_mag = 2 * np.pi / lamda_v
    kf_vec = Q_vec_v + k_mag * ki_hat[:, None]  # (3, M_v)
    kf_dir = kf_vec / np.linalg.norm(kf_vec, axis=0, keepdims=True)  # (3, M_v)

    # 4. Resolve the True Sample Ray Origin (s_lab)
    if gonio_axes is not None and gonio_angles is not None:
        from subhkl.instrument.goniometer import sample_to_lab
        offsets = sample_offset
        if offsets is not None and offsets.ndim == 1:
            offsets_full = np.zeros((len(gonio_axes), 3))
            offsets_full[-1] = offsets
        elif offsets is None:
            offsets_full = np.zeros((len(gonio_axes), 3))
        else:
            offsets_full = offsets

        s_lab = sample_to_lab(
            np.array([0.0, 0.0, 0.0]), 
            gonio_axes, 
            gonio_angles, 
            offsets_full, 
            zero_offsets=gonio_offsets
        )
    else:
        # Legacy fallback
        s = sample_offset if sample_offset is not None else np.zeros(3)
        if R_all is not None:
            s_lab = R_all @ s
        else:
            s_lab = s

    # 5. Intersect with Panel
    mask_panel, row, col = detector.reflections_mask(
        kf_dir[0], kf_dir[1], kf_dir[2], sample_offset=s_lab
    )

    return (
        row[mask_panel], col[mask_panel],
        h_v[mask_panel], k_v[mask_panel], l_v[mask_panel],
        lamda_v[mask_panel],
    )


def calculate_angular_error(
    xyz_det: npt.NDArray,
    h: npt.NDArray,
    k: npt.NDArray,
    l: npt.NDArray,  # noqa: E741
    lam: npt.NDArray,
    RUB: npt.NDArray,
    sample_offset: npt.NDArray = None,
    ki_vec: npt.NDArray = None,
    R_all: npt.NDArray = None,
    gonio_axes: npt.NDArray = None,
    gonio_angles: npt.NDArray = None,
):
    """
    Calculate D-spacing and Angular errors for observed peaks vs predicted geometry.
    Uses the RUB matrix (R @ U @ B) for all coordinate transformations.
    gonio_angles is expected to be shape (N_peaks, N_axes).
    """
    if sample_offset is None:
        sample_offset = np.zeros(3)
    if ki_vec is None:
        ki_vec = np.array([0.0, 0.0, 1.0])

    # 1. Calculate Q_calc (Lab Frame)
    q_lab_calc = get_q_lab(h, k, l, RUB)
    q_calc_norm = q_lab_calc / np.linalg.norm(q_lab_calc, axis=1, keepdims=True)

    # 2. Calculate Q_obs (Lab Frame) from Detector Pixel Position
    if gonio_axes is not None and gonio_angles is not None:
        offsets = sample_offset
        if offsets.ndim == 1:
            offsets_full = np.zeros((len(gonio_axes), 3))
            offsets_full[-1] = offsets
        else:
            offsets_full = offsets

        s_lab = np.zeros((len(xyz_det), 3))
        deg2rad = np.pi / 180.0

        angles = np.tile(gonio_angles, (len(xyz_det), 1)) if gonio_angles.ndim == 1 else gonio_angles

        for i in reversed(range(len(gonio_axes))):
            direction = gonio_axes[i][:3]
            direction_mult = gonio_axes[i][3] if len(gonio_axes[i]) > 3 else 1.0
            direction = direction / np.linalg.norm(direction)

            # Apply zero point to true angle
            axis_offset = gonio_offsets[i] if gonio_offsets is not None else 0.0
            true_angle = angles[:, i] + axis_offset
            theta = direction_mult * true_angle * deg2rad 

            K = np.array([
                [0, -direction[2], direction[1]],
                [direction[2], 0, -direction[0]],
                [-direction[1], direction[0], 0]
            ])

            sin_t = np.sin(theta)[:, None, None]
            cos_t = np.cos(theta)[:, None, None]
            K_sq = K @ K
            R_i = np.eye(3)[None, :, :] + sin_t * K[None, :, :] + (1 - cos_t) * K_sq[None, :, :]

            s_lab = np.einsum("nij,nj->ni", R_i, s_lab + offsets_full[i])

        v = xyz_det - s_lab
    elif R_all is not None:
        # Legacy static fallback
        if R_all.ndim == 3:
            s_lab = np.einsum("nij,j->ni", R_all, sample_offset)
        else:
            s_lab = R_all @ sample_offset
        v = xyz_det - s_lab
    else:
        v = xyz_det - sample_offset
    # ------------------------------------------------------

    dist = np.linalg.norm(v, axis=1, keepdims=True)
    kf_dir = v / dist  # Unit vector pointing from sample to pixel

    ki_dir = ki_vec / np.linalg.norm(ki_vec)

    # Scattering vector direction matches delta_k = kf - ki
    delta_k = kf_dir - ki_dir[None, :]
    two_sin_theta = np.linalg.norm(delta_k, axis=1)

    # Q_obs direction
    with np.errstate(divide="ignore", invalid="ignore"):
        q_obs_norm = delta_k / two_sin_theta[:, None]
        q_obs_norm = np.nan_to_num(q_obs_norm, nan=0.0)

    # 3. Angular Error (Angle between Q_calc direction and Q_obs direction)
    dot = np.sum(q_obs_norm * q_calc_norm, axis=1)
    dot = np.clip(dot, -1.0, 1.0)
    ang_err = np.rad2deg(np.arccos(dot))

    # 4. D-Spacing Error
    with np.errstate(divide="ignore", invalid="ignore"):
        d_obs = np.divide(lam, two_sin_theta)

    q_lab_mag = np.linalg.norm(q_lab_calc, axis=1)

    with np.errstate(divide="ignore", invalid="ignore"):
        d_calc = np.divide(1.0, q_lab_mag)

    d_err = np.abs(d_obs - d_calc)

    return d_err, ang_err
