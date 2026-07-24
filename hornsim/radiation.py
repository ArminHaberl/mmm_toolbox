"""Radiation impedance and far-field pressure calculation.

MATLAB originals:
  - MMM_ASbaffledradzmatrixIntp -> baffled_rad_zmatrix_axi
  - MMM_ASbaffledradzmatrix -> baffled_rad_zmatrix_direct_axi
  - MMM_ASbaffledradzmatrixPrecompute -> precompute_rad_zmatrix
  - MMM_ASradiatedPressure -> radiated_pressure_axi
"""

import time
from pathlib import Path

import numpy as np
from scipy.integrate import quad
from scipy.interpolate import CubicSpline
from scipy.io import loadmat, savemat
from scipy.special import j0, j1


def _struve_h1(x: np.ndarray) -> np.ndarray:
    """Struve function H1(x) — MathWorld approximation (matches MATLAB)."""
    return (
        2.0 / np.pi
        - j0(x)
        + (16.0 / np.pi - 5.0) * np.sin(x) / x
        + (12.0 - 36.0 / np.pi) * (1.0 - np.cos(x)) / x**2
    )


# ---------------------------------------------------------------------------
# Direct numerical integration helpers
# ---------------------------------------------------------------------------


def _func_dn(
    tau: float, gamma_n: float, kR: float
) -> float:
    """Core integrand kernel: D_n(tau) = -√2·τ·J₁(τ·kR) / ((γₙ/kR)² - τ²)."""
    return (
        -np.sqrt(2.0)
        * tau
        * j1(tau * kR)
        / ((gamma_n / kR) ** 2 - tau**2)
    )


def _resistance_integrand(
    phi: float, n: int, m: int, kR: float, bz: np.ndarray
) -> float:
    """Integrand for modal radiation resistance (φ ∈ [0, π/2])."""
    sinphi = np.sin(phi)
    return sinphi * _func_dn(sinphi, bz[n], kR) * _func_dn(sinphi, bz[m], kR)


def _reactance_integrand(
    phi: float, n: int, m: int, kR: float, bz: np.ndarray
) -> float:
    """Integrand for modal radiation reactance (φ ∈ [0, 10])."""
    coshphi = np.cosh(phi)
    return (
        coshphi
        * _func_dn(coshphi, bz[n], kR)
        * _func_dn(coshphi, bz[m], kR)
    )


# ---------------------------------------------------------------------------
# Direct numerical integration (slow but reference-quality)
# ---------------------------------------------------------------------------


def baffled_rad_zmatrix_direct_axi(
    k: np.ndarray,
    rho: float,
    c: float,
    S: float,
    max_modes: int,
    bz: np.ndarray,
    use_hf_approx: bool = False,
    progress_report: bool = False,
) -> np.ndarray:
    """Modal radiation impedance via direct numerical integration.

    Corresponds to MMM_ASbaffledradzmatrix.

    Integrates over φ ∈ [0, π/2] for resistance and φ ∈ [0, 10] for
    reactance using adaptive Gauss-Kronrod quadrature. Substantially
    slower than the interpolation version; intended for precomputation
    of lookup tables.

    Parameters
    ----------
    k : (nfreq,) array
        Wavenumbers [rad/m].
    rho : float
        Density of medium [kg/m³].
    c : float
        Sound speed [m/s].
    S : float
        Cross-section area of opening [m²].
    max_modes : int
        Number of modes.
    bz : (max_modes,) array
        Zeros of J₁ (eigenvalues).
    use_hf_approx : bool
        If True, use asymptotic formulas above 2.5× the mode cutoff
        frequency instead of numerical integration.
    progress_report : bool
        Print progress.

    Returns
    -------
    Zmat : (max_modes, max_modes, nfreq) complex ndarray
        Modal radiation impedance matrix.
    """
    nfreq = len(k)
    a = np.sqrt(S / np.pi)
    kR = k * a

    tol = np.maximum(
        np.minimum(0.01, 10.0 ** (-np.log10(kR) - 1)), 1e-6
    )
    if nfreq > 1:
        tol = np.full_like(kR, 1e-6)

    # Analytical fundamental mode
    R00 = 1.0 - j1(2.0 * kR) / kR
    X00 = 2.0 * _struve_h1(2.0 * kR) / (2.0 * kR)

    Zmat = np.zeros((max_modes, max_modes, nfreq), dtype=complex)

    n_total = max_modes * max_modes
    count = 0
    t_start = time.perf_counter()

    for m in range(max_modes):
        for n in range(max_modes):
            if m == 0 and n == 0:
                Z = R00 + 1j * X00
                count += 1
            elif n >= m:
                count += 1
                mu_max = max(bz[m], bz[n])

                if use_hf_approx:
                    int_id = np.where(kR < (mu_max * 2.5 + 1))[0]
                    int_hf = np.where(kR >= mu_max * 2.5)[0]
                else:
                    int_id = np.arange(nfreq)
                    int_hf = np.array([], dtype=int)

                R = np.zeros(nfreq)
                X = np.zeros(nfreq)

                # High-frequency asymptotic
                if len(int_hf) > 0:
                    if m == n:
                        R[int_hf] = (
                            R00[int_hf]
                            * k[int_hf]
                            / np.sqrt(
                                k[int_hf] ** 2 - (mu_max / a) ** 2
                            )
                        )
                    else:
                        R[int_hf] = (R00[int_hf] - 1.0) / (
                            1.0 - (mu_max / kR[int_hf]) ** 2
                        )
                    X[int_hf] = X00[int_hf] / (
                        1.0 - (mu_max / kR[int_hf]) ** 2
                    )

                # Numerical integration for remaining points
                for nk in int_id:
                    R[nk], _ = quad(
                        _resistance_integrand,
                        0.0,
                        np.pi / 2.0,
                        args=(m, n, kR[nk], bz),
                        epsabs=tol[nk],
                        limit=500,
                    )
                    X[nk], _ = quad(
                        _reactance_integrand,
                        0.0,
                        10.0,
                        args=(m, n, kR[nk], bz),
                        epsabs=1e-6,
                        limit=500,
                    )

                Z = R + 1j * X
            else:
                Z = Zmat[n, m, :]
                count += 1

            Zmat[m, n, :] = Z

            if progress_report and n >= m:
                pct = 100.0 * count / n_total
                elapsed = time.perf_counter() - t_start
                print(
                    f"{pct:5.1f}%: mode ({m},{n}) "
                    f"({elapsed:.1f}s elapsed)"
                )

    Zmat = rho * c / S * Zmat
    return Zmat


# ---------------------------------------------------------------------------
# Precomputation of radiation impedance lookup table
# ---------------------------------------------------------------------------


def precompute_rad_zmatrix(
    max_modes: int = 32,
    bessel_zeros_path: str = "matlab/MMM_besselzeros.mat",
    output_path: str | None = None,
    progress_report: bool = True,
) -> str:
    """Precompute and save a modal radiation impedance lookup table.

    Corresponds to MMM_ASbaffledradzmatrixPrecompute.

    Generates a ka grid (20 log-space points below ka=3, then linearly
    spaced to 2.5× the highest modal cutoff), computes the normalized
    radiation impedance matrix via direct numerical integration, and
    saves the result as a .mat file compatible with
    ``baffled_rad_zmatrix_axi``.

    Parameters
    ----------
    max_modes : int
        Number of modes to precompute (default 32).
    bessel_zeros_path : str
        Path to MMM_besselzeros.mat.
    output_path : str or None
        Output .mat file path. Defaults to
        ``matlab/ZradAS{max_modes}.mat``.
    progress_report : bool
        Print progress.

    Returns
    -------
    output_path : str
        Path to the saved file.
    """
    bz_mat = loadmat(bessel_zeros_path)
    bz = bz_mat["bz"].flatten()

    if output_path is None:
        output_path = f"matlab/ZradAS{max_modes}.mat"

    kamax = bz[max_modes - 1] * 2.5
    kamin = 0.1
    nk = int(bz[max_modes - 1] * 4)

    ka1 = np.logspace(np.log10(kamin), np.log10(3.0), 20)
    ka2 = np.linspace(3.0, kamax, nk)
    ka = np.concatenate([ka1, ka2[1:]])
    k = ka * np.sqrt(np.pi)

    print(
        f"Precomputing radiation impedance for {max_modes} modes "
        f"({nk + 19} ka points)..."
    )
    print(
        "This may take several minutes."
    )

    t_start = time.perf_counter()
    Zmat = baffled_rad_zmatrix_direct_axi(
        k, 1.0, 1.0, 1.0, max_modes, bz,
        use_hf_approx=True, progress_report=progress_report,
    )
    elapsed = time.perf_counter() - t_start

    savemat(output_path, {"ka": ka, "Zmat": Zmat})
    print(f"Saved to {output_path} ({elapsed:.1f}s)")

    return output_path


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
