"""Radiation impedance and far-field pressure calculation.

MATLAB originals:
  - MMM_ASbaffledradzmatrixIntp -> baffled_rad_zmatrix_axi
  - MMM_ASradiatedPressure -> radiated_pressure_axi
"""

import numpy as np
from scipy.interpolate import CubicSpline
from scipy.io import loadmat
from scipy.special import j0, j1


def _struve_h1(x: np.ndarray) -> np.ndarray:
    """Struve function H1(x) — MathWorld approximation (matches MATLAB)."""
    return (
        2.0 / np.pi
        - j0(x)
        + (16.0 / np.pi - 5.0) * np.sin(x) / x
        + (12.0 - 36.0 / np.pi) * (1.0 - np.cos(x)) / x**2
    )


def baffled_rad_zmatrix_axi(
    k: np.ndarray,
    rho: float,
    c: float,
    S: float,
    max_modes: int,
    filename: str,
) -> np.ndarray:
    """Modal radiation impedance matrix for circular aperture in infinite baffle.

    Corresponds to MMM_ASbaffledradzmatrixIntp.
    Uses interpolation from a precomputed lookup table.

    Returns (max_modes, max_modes, len(k)) complex array.
    """
    zrad_mat = loadmat(filename)
    ka = zrad_mat["ka"].flatten()
    Zmat = zrad_mat["Zmat"]

    available_modes = Zmat.shape[0]
    if max_modes > available_modes:
        raise ValueError(
            f"Higher number of modes requested ({max_modes}) than "
            f"precalculated ({available_modes})"
        )

    # Load Bessel zeros from the same directory as the Zrad file
    bz_mat = loadmat("matlab/MMM_besselzeros.mat")
    bz = bz_mat["bz"].flatten()

    a = np.sqrt(S / np.pi)
    kain = k * a

    # Analytical fundamental mode (0,0)
    R00 = 1.0 - j1(2.0 * kain) / kain
    X00 = 2.0 * _struve_h1(2.0 * kain) / (2.0 * kain)

    nfreq = len(k)
    ZmatOut = np.zeros((max_modes, max_modes, nfreq), dtype=complex)

    # Precompute interpolation masks
    ka_min = np.min(ka)
    ka_max = np.max(ka)
    intp_id = np.where((kain >= ka_min) & (kain <= ka_max))[0]
    pos_id = np.where(kain > ka_max)[0]

    minka = 1.0
    intp_id_r = np.where((kain >= minka) & (kain <= ka_max))[0]

    for m in range(max_modes):
        for n in range(max_modes):
            if m == 0 and n == 0:
                Zmn = R00 + 1j * X00
            elif m <= n:
                bzq = max(bz[m], bz[n])

                # --- Resistance ---
                Y = np.real(Zmat[m, n, :])
                zp = _get_poly_coeff(m + 1, n + 1, bz)

                # R1: low-frequency polynomial extrapolation (kain < 1)
                low_id = np.where(kain < minka)[0]
                R1 = np.polyval(zp, kain[low_id] ** 2)

                # R2: mid-frequency spline interpolation
                if len(intp_id_r) > 0:
                    cs_r = CubicSpline(ka, Y)
                    R2 = cs_r(kain[intp_id_r])
                else:
                    R2 = np.array([])

                # R3: high-frequency asymptotic
                if m == n:
                    R3 = (
                        R00[pos_id]
                        * k[pos_id]
                        / np.sqrt(k[pos_id] ** 2 - (bz[m] / a) ** 2)
                    )
                else:
                    R3 = (R00[pos_id] - 1.0) / (
                        1.0 - (bzq / kain[pos_id]) ** 2
                    )

                R = np.concatenate([R1, R2, R3])

                # --- Reactance ---
                Y = np.real(Zmat[m, n, :])

                # X1: low-frequency linear extrapolation (kain < ka(1))
                low_x_id = np.where(kain < ka[0])[0]
                X1 = Y[0] * kain[low_x_id] / ka[0]

                # X2: mid-frequency spline interpolation
                Y_x = np.imag(Zmat[m, n, :])
                if len(intp_id) > 0:
                    cs_x = CubicSpline(ka, Y_x)
                    X2 = cs_x(kain[intp_id])
                else:
                    X2 = np.array([])

                # X3: high-frequency asymptotic
                X3 = X00[pos_id] / (1.0 - (bzq / kain[pos_id]) ** 2)

                X = np.concatenate([X1, X2, X3])
                Zmn = R + 1j * X
            else:
                Zmn = ZmatOut[n, m, :]

            ZmatOut[m, n, :] = Zmn

    ZmatOut = rho * c / S * ZmatOut
    return ZmatOut


def _get_poly_coeff(n: int, m: int, bz: np.ndarray) -> np.ndarray:
    """Compute polynomial coefficients for low-frequency R extrapolation."""
    nlow = min(n, m)
    nhigh = max(n, m)
    if nlow == 1 and nhigh > 1:
        idx = nhigh - 1  # convert to 0-based
        p = np.zeros(4)
        p[1] = -1.0 / (3.0 * bz[idx] ** 2)
        p[0] = (
            -4.0 / (15.0 * bz[idx] ** 4) + 1.0 / (15.0 * bz[idx] ** 2)
        )
    else:
        idx_m = m - 1
        idx_n = n - 1
        p = np.zeros(5)
        p[1] = 4.0 / (15.0 * bz[idx_m] ** 2 * bz[idx_n] ** 2)
        p[0] = (
            8.0 / (35.0 * bz[idx_m] ** 4 * bz[idx_n] ** 2)
            + 8.0 / (35.0 * bz[idx_m] ** 2 * bz[idx_n] ** 4)
            - 2.0 / (35.0 * bz[idx_m] ** 2 * bz[idx_n] ** 2)
        )
    return p


def radiated_pressure_axi(
    data: dict,
    field_points: np.ndarray,
    use_farfield_approx: bool = True,
) -> dict:
    """Calculate radiated pressure at given field points.

    Corresponds to MMM_ASradiatedPressure.

    Modifies data dict in-place: sets data['pRad'].
    Returns data for convenience.
    """
    if use_farfield_approx:
        data["pRad"] = _modal_radiated_pressure(data, field_points)
    else:
        # Rayleigh integral branch — not covered by current tests
        raise NotImplementedError(
            "Rayleigh integral method not yet ported."
        )

    return data


def _modal_radiated_pressure(
    data: dict, field_points: np.ndarray
) -> np.ndarray:
    """Far-field modal radiated pressure calculation."""
    a = np.sqrt(data["Sm"] / np.pi)
    nfreq = data["nfreq"]
    n_modes = data["n_modes"]
    n_points = field_points.shape[0]
    prext = np.zeros((n_points, nfreq), dtype=complex)

    for ii in range(n_points):
        pe = field_points[ii, :]
        R = np.linalg.norm(pe)
        theta = np.arctan2(pe[0], pe[1])
        s = data["k"] * a * np.sin(theta)

        sm = s[np.newaxis, :]  # (1, nfreq)
        bzm = data["eigen_values"][:n_modes, np.newaxis]  # (n_modes, 1)

        Theta2M = 2.0 * sm * j1(sm) / (sm**2 - bzm**2)

        if theta == 0.0:
            Theta2M[0, :] = 1.0

        if n_modes > 1:
            ModalSum = np.conj(
                np.sum(1j * (Theta2M * data["Umouth"]), axis=0)
            )
        else:
            ModalSum = 1j * Theta2M.flatten()

        pf = (
            data["rho"]
            * data["c"]
            / (2.0 * np.pi * R)
            * np.exp(-1j * data["k"] * R)
            * data["k"]
        )

        prext[ii, :] = pf * ModalSum

    return prext
