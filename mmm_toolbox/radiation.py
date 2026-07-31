"""Radiation impedance and far-field pressure calculation.

MATLAB originals:
  - MMM_ASbaffledradzmatrixIntp -> baffled_rad_zmatrix_axi
  - MMM_ASbaffledradzmatrix -> baffled_rad_zmatrix_direct_axi
  - MMM_ASradiatedPressure -> radiated_pressure_axi
  - MMM_ASpressureDistribution -> pressure_distribution_axi
"""

import time
from pathlib import Path

import numpy as np
from scipy.interpolate import CubicSpline
from scipy.io import loadmat, savemat
from scipy.special import j0, j1, roots_legendre

from mmm_toolbox.axi import _get_bessel_zeros, get_eigenfunctions_axi


def _struve_h1(x: np.ndarray) -> np.ndarray:
    """Struve function H1(x) — MathWorld approximation (matches MATLAB)."""
    return (
        2.0 / np.pi
        - j0(x)
        + (16.0 / np.pi - 5.0) * np.sin(x) / x
        + (12.0 - 36.0 / np.pi) * (1.0 - np.cos(x)) / x**2
    )


# ---------------------------------------------------------------------------
# Fixed Gauss-Legendre quadrature (core numerical integration)
# ---------------------------------------------------------------------------


def _compute_zmat_fixed_quad(
    k: np.ndarray,
    kR: np.ndarray,
    a: float,
    bz: np.ndarray,
    max_modes: int,
    use_hf_approx: bool,
    n_quad: int = 500,
    progress_report: bool = False,
) -> np.ndarray:
    """Compute entire Zmat via fixed Gauss-Legendre quadrature + broadcasting.

    Avoids per-mode-pair loops by integrating all modes simultaneously
    at each quadrature node.  The (0,0) mode is overwritten with the
    analytical closed-form solution after quadrature.
    """
    nfreq = len(k)
    M = max_modes

    R00 = 1.0 - j1(2.0 * kR) / kR
    X00 = 2.0 * _struve_h1(2.0 * kR) / (2.0 * kR)

    Zmat = np.zeros((M, M, nfreq), dtype=complex)

    bz_div_kR_sq = (bz[:M, np.newaxis] / kR[np.newaxis, :]) ** 2
    limit_value = np.sqrt(2.0) / 2.0 * kR[np.newaxis, :] * j0(bz[:M, np.newaxis])

    # ---- Resistance: phi in [0, pi/2] ----
    nodes, weights = roots_legendre(n_quad)
    t = 0.5 * (nodes + 1.0) * (np.pi / 2.0)
    w = weights * (np.pi / 4.0)

    for idx in range(n_quad):
        sinphi = np.sin(t[idx])
        tau = sinphi
        if tau == 0.0:
            continue

        Jv = j1(tau * kR)
        denom = bz_div_kR_sq - tau**2
        D = -np.sqrt(2.0) * tau * Jv[np.newaxis, :] / denom

        D = np.where(np.abs(denom) < 1e-12, limit_value, D)

        Zmat.real += w[idx] * sinphi * D[:, np.newaxis, :] * D[np.newaxis, :, :]

        if progress_report and (idx + 1) % 100 == 0:
            pct = 100.0 * (idx + 1) / n_quad
            print(f"  Resistance quadrature: {pct:.0f}%")

    Zmat.real[0, 0, :] = R00

    # ---- Reactance: phi in [0, 10] ----
    nodes_x, weights_x = roots_legendre(n_quad)
    t_x = 0.5 * (nodes_x + 1.0) * 10.0
    w_x = weights_x * 5.0

    for idx in range(n_quad):
        coshphi = np.cosh(t_x[idx])
        tau = coshphi

        Jv = j1(tau * kR)
        denom = bz_div_kR_sq - tau**2
        D = -np.sqrt(2.0) * tau * Jv[np.newaxis, :] / denom
        D = np.where(np.abs(denom) < 1e-12, limit_value, D)

        Zmat.imag += w_x[idx] * coshphi * D[:, np.newaxis, :] * D[np.newaxis, :, :]

        if progress_report and (idx + 1) % 100 == 0:
            pct = 100.0 * (idx + 1) / n_quad
            print(f"  Reactance quadrature: {pct:.0f}%")

    Zmat.imag[0, 0, :] = X00

    # ---- High-frequency asymptotic ---
    if use_hf_approx:
        bz_M = bz[:M]
        mu_max_ij = np.maximum(bz_M[:, np.newaxis], bz_M[np.newaxis, :])
        hf_mask = kR[np.newaxis, np.newaxis, :] >= mu_max_ij[:, :, np.newaxis] * 2.5
        hf_mask[0, 0, :] = False

        if np.any(hf_mask):
            X_hf = X00[np.newaxis, np.newaxis, :] / (
                1.0 - (mu_max_ij[:, :, np.newaxis] / kR[np.newaxis, np.newaxis, :]) ** 2
            )
            R_hf_off = (R00[np.newaxis, np.newaxis, :] - 1.0) / (
                1.0 - (mu_max_ij[:, :, np.newaxis] / kR[np.newaxis, np.newaxis, :]) ** 2
            )

            diag = np.arange(M)
            mu_diag = bz_M[diag]
            hf_diag = hf_mask[diag, diag, :]
            R_hf_diag = (
                R00[np.newaxis, :]
                * k[np.newaxis, :]
                / np.sqrt(np.maximum(
                    k[np.newaxis, :] ** 2 - (mu_diag[:, np.newaxis] / a) ** 2, 1e-300,
                ))
            )
            X_hf_diag = X00[np.newaxis, :] / (
                1.0 - (mu_diag[:, np.newaxis] / kR[np.newaxis, :]) ** 2
            )

            Zmat.real = np.where(hf_mask, R_hf_off, Zmat.real)
            Zmat.imag = np.where(hf_mask, X_hf, Zmat.imag)
            Zmat.real[diag, diag, :] = np.where(hf_diag, R_hf_diag, Zmat.real[diag, diag, :])
            Zmat.imag[diag, diag, :] = np.where(hf_diag, X_hf_diag, Zmat.imag[diag, diag, :])

    return Zmat


# ---------------------------------------------------------------------------
# Public API -- direct numerical integration
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
    n_quad: int = 500,
) -> np.ndarray:
    """Modal radiation impedance via fixed Gauss-Legendre quadrature.

    Corresponds to MMM_ASbaffledradzmatrix.

    Integrates over phi in [0, pi/2] for resistance and phi in [0, 10]
    for reactance, broadcasting over all mode pairs simultaneously at
    each quadrature node.

    Parameters
    ----------
    k : (nfreq,) array
        Wavenumbers [rad/m].
    rho : float
        Density of medium [kg/m^3].
    c : float
        Sound speed [m/s].
    S : float
        Cross-section area of opening [m^2].
    max_modes : int
        Number of modes.
    bz : (max_modes,) array
        Zeros of J1 (eigenvalues).
    use_hf_approx : bool
        If True, use asymptotic formulas above 2.5x the mode cutoff
        frequency instead of numerical integration.
    progress_report : bool
        Print progress.
    n_quad : int
        Number of Gauss-Legendre nodes (applied to both R and X).

    Returns
    -------
    Zmat : (max_modes, max_modes, nfreq) complex ndarray
        Modal radiation impedance matrix.
    """
    a = np.sqrt(S / np.pi)
    kR = k * a

    t_start = time.perf_counter()
    Zmat = _compute_zmat_fixed_quad(
        k, kR, a, bz, max_modes, use_hf_approx,
        n_quad=n_quad,
        progress_report=progress_report,
    )
    elapsed = time.perf_counter() - t_start
    if progress_report:
        print(f"Fixed-quadrature done ({elapsed:.1f}s)")

    return rho * c / S * Zmat


# ---------------------------------------------------------------------------
# Lookup table cache (disk-backed, ~/.cache/mmm_toolbox/)
# ---------------------------------------------------------------------------


def _build_lookup_table(
    max_modes: int, n_quad: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute the normalized (rho=1, c=1, S=1) lookup table from scratch."""
    bz = _get_bessel_zeros(max_modes)

    kamax = bz[max_modes - 1] * 2.5
    kamin = 0.1
    nk = int(bz[max_modes - 1] * 4.3)

    ka1 = np.logspace(np.log10(kamin), np.log10(3.0), 20)
    ka2 = np.linspace(3.0, kamax, nk)
    ka = np.concatenate([ka1, ka2[1:]])
    k = ka * np.sqrt(np.pi)
    a = np.sqrt(1.0 / np.pi)
    kR = k * a

    Zmat = _compute_zmat_fixed_quad(
        k, kR, a, bz, max_modes, use_hf_approx=True,
        n_quad=n_quad,
    )
    return ka, Zmat


def _get_lookup_table(
    max_modes: int, n_quad: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (ka, Zmat) from disk cache, or build + cache if missing.

    Scans for any cached table with at least *max_modes* and matching
    *n_quad* before building a new one, so that an 8-mode request
    reuses an existing 32-mode table instead of building from scratch.
    """
    cache_dir = Path.home() / ".cache" / "mmm_toolbox"
    cache_dir.mkdir(parents=True, exist_ok=True)

    # Reuse any cached table with >= max_modes and matching n_quad
    for cf in sorted(cache_dir.glob(f"ZradAS*_q{n_quad}.mat")):
        try:
            cached = int(cf.stem.split("_")[0].removeprefix("ZradAS"))
        except (ValueError, IndexError):
            continue
        if cached >= max_modes:
            data = loadmat(str(cf))
            return data["ka"].flatten(), data["Zmat"]

    cache_file = cache_dir / f"ZradAS{max_modes}_q{n_quad}.mat"

    print(
        f"Building radiation-impedance lookup table ({max_modes} modes, "
        f"{n_quad} quadrature nodes)..."
    )
    ka, Zmat = _build_lookup_table(max_modes, n_quad)
    savemat(str(cache_file), {"ka": ka, "Zmat": Zmat})
    print(f"Cached to {cache_file}")
    return ka, Zmat


# ---------------------------------------------------------------------------
# Public API -- interpolation from lookup table
# ---------------------------------------------------------------------------


def baffled_rad_zmatrix_axi(
    k: np.ndarray,
    rho: float,
    c: float,
    S: float,
    max_modes: int,
    filename: str | None = None,
    n_quad: int = 2000,
) -> np.ndarray:
    """Modal radiation impedance matrix for circular aperture in infinite baffle.

    Corresponds to MMM_ASbaffledradzmatrixIntp.

    Uses cubic-spline interpolation from a precomputed lookup table.
    By default the table is built on first call and cached to disk at
    ~/.cache/mmm_toolbox/.  Pass *filename* to load a custom
    precomputed .mat file instead.

    Returns (max_modes, max_modes, len(k)) complex array.
    """
    if filename is not None:
        zrad_mat = loadmat(filename)
        ka = zrad_mat["ka"].flatten()
        Zmat = zrad_mat["Zmat"]
    else:
        ka, Zmat = _get_lookup_table(max_modes, n_quad)

    available_modes = Zmat.shape[0]
    if max_modes > available_modes:
        raise ValueError(
            f"Higher number of modes requested ({max_modes}) than "
            f"precalculated ({available_modes})"
        )

    bz = _get_bessel_zeros(max_modes)

    a = np.sqrt(S / np.pi)
    kain = k * a

    R00 = 1.0 - j1(2.0 * kain) / kain
    X00 = 2.0 * _struve_h1(2.0 * kain) / (2.0 * kain)

    nfreq = len(k)
    ZmatOut_real = np.zeros((max_modes, max_modes, nfreq))
    ZmatOut_imag = np.zeros((max_modes, max_modes, nfreq))

    ka_min = np.min(ka)
    ka_max = np.max(ka)
    intp_id = np.where((kain >= ka_min) & (kain <= ka_max))[0]
    pos_id = np.where(kain > ka_max)[0]
    low_id = np.where(kain < 1.0)[0]
    low_x_id = np.where(kain < ka[0])[0]

    minka = 1.0
    intp_id_r = np.where((kain >= minka) & (kain <= ka_max))[0]

    for m in range(max_modes):
        for n in range(m, max_modes):
            if m == 0 and n == 0:
                ZmatOut_real[0, 0, :] = R00
                ZmatOut_imag[0, 0, :] = X00
                continue

            bzq = max(bz[m], bz[n])

            zp = _get_poly_coeff(m + 1, n + 1, bz)
            if len(low_id) > 0:
                ZmatOut_real[m, n, low_id] = np.polyval(zp, kain[low_id] ** 2)

            if len(intp_id_r) > 0:
                Y_r = np.real(Zmat[m, n, :])
                cs_r = CubicSpline(ka, Y_r)
                ZmatOut_real[m, n, intp_id_r] = cs_r(kain[intp_id_r])

            if len(pos_id) > 0:
                if m == n:
                    ZmatOut_real[m, n, pos_id] = (
                        R00[pos_id]
                        * k[pos_id]
                        / np.sqrt(k[pos_id] ** 2 - (bz[m] / a) ** 2)
                    )
                else:
                    ZmatOut_real[m, n, pos_id] = (R00[pos_id] - 1.0) / (
                        1.0 - (bzq / kain[pos_id]) ** 2
                    )

            Y0_r = np.real(Zmat[m, n, 0])
            if len(low_x_id) > 0:
                ZmatOut_imag[m, n, low_x_id] = Y0_r * kain[low_x_id] / ka[0]

            if len(intp_id) > 0:
                Y_x = np.imag(Zmat[m, n, :])
                cs_x = CubicSpline(ka, Y_x)
                ZmatOut_imag[m, n, intp_id] = cs_x(kain[intp_id])

            if len(pos_id) > 0:
                ZmatOut_imag[m, n, pos_id] = X00[pos_id] / (
                    1.0 - (bzq / kain[pos_id]) ** 2
                )

            if m != n:
                ZmatOut_real[n, m, :] = ZmatOut_real[m, n, :]
                ZmatOut_imag[n, m, :] = ZmatOut_imag[m, n, :]

    ZmatOut = ZmatOut_real + 1j * ZmatOut_imag
    ZmatOut = rho * c / S * ZmatOut
    return ZmatOut


def _get_poly_coeff(n: int, m: int, bz: np.ndarray) -> np.ndarray:
    """Compute polynomial coefficients for low-frequency R extrapolation."""
    nlow = min(n, m)
    nhigh = max(n, m)
    if nlow == 1 and nhigh > 1:
        idx = nhigh - 1
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
        data["pRad"] = _rayleigh_radiated_pressure(data, field_points)

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

        sm = s[np.newaxis, :]
        bzm = data["eigen_values"][:n_modes, np.newaxis]

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


# ---------------------------------------------------------------------------
# Rayleigh integral (near-field)
# ---------------------------------------------------------------------------


def _rayleigh_radiated_pressure(
    data: dict, field_points: np.ndarray
) -> np.ndarray:
    """Near-field radiated pressure via Rayleigh integral over the mouth.

    Corresponds to rayleighRadiatedPressure (nested in
    MMM_ASradiatedPressure).
    """
    a = np.sqrt(data["Sm"] / np.pi)
    n_int = data["n_integration_points"]

    r = np.linspace(0.0, a, n_int)
    rp1 = r[:-1]
    rp2 = r[1:]
    rp = (rp1 + rp2) / 2.0

    phi = get_eigenfunctions_axi(a, rp, data["eigen_values"], normalize=True)
    Ur = phi @ data["Umouth"]
    if Ur.ndim == 1:
        Ur = Ur[:, np.newaxis]
    uo = Ur / (a**2 * np.pi)

    n_points = field_points.shape[0]
    nfreq = data["nfreq"]
    prext = np.zeros((n_points, nfreq), dtype=complex)

    for ii in range(n_points):
        point = field_points[ii, :]
        prext[ii, :] = _rayleigh_integral(
            data["k"], r, uo, point, data["rho"], data["c"]
        )

    return prext


def _rayleigh_integral(
    k: np.ndarray,
    vert: np.ndarray,
    vvel: np.ndarray,
    pe: np.ndarray,
    rho: float,
    c: float,
) -> np.ndarray:
    """Rayleigh integral over annular rings for a single field point.

    Corresponds to MMM_ASrayleighint.
    """
    p = np.array([pe[0], 0.0, pe[1]])
    NV = len(vert)
    nk = len(k)

    QA = vert[:-1]
    QB = vert[1:]
    radmid = (QA + QB) / 2.0
    glen = np.abs(QA - QB)
    cirmid = 2.0 * np.pi * radmid
    NT = np.ceil(1.0 + cirmid / glen).astype(int)

    phiext = np.zeros(nk, dtype=complex)

    for ir in range(NV - 1):
        NTi = NT[ir]
        theta = np.arange(NTi) * 2.0 * np.pi / NTi
        dtheta = 2.0 * np.pi / NTi
        dS = 0.5 * dtheta * (QB[ir] ** 2 - QA[ir] ** 2)

        q = np.zeros((3, NTi))
        q[0, :] = radmid[ir] * np.cos(theta)
        q[1, :] = radmid[ir] * np.sin(theta)
        q[2, :] = 0.0

        r = q - p[:, np.newaxis]
        r = np.sqrt(r[0, :] ** 2 + r[1, :] ** 2 + r[2, :] ** 2)

        vv = np.atleast_1d(vvel[ir])
        phiext = phiext + np.sum(
            vv.reshape(1, -1) * dS
            * np.exp(-1j * r[:, np.newaxis] * k[np.newaxis, :])
            / r[:, np.newaxis],
            axis=0,
        )

    prext = 1j * k * rho * c / (2.0 * np.pi) * phiext
    return prext


# ---------------------------------------------------------------------------
# Pressure distribution
# ---------------------------------------------------------------------------


def pressure_distribution_axi(
    freq: float,
    data: dict,
    add_nearfield: bool = False,
    resolution: int = 30,
):
    """Compute spatial pressure field inside (and in front of) a horn.

    Corresponds to MMM_ASpressureDistribution.

    Parameters
    ----------
    freq : float
        Single frequency [Hz].
    data : dict
        Initialized horn data dict (from init_horn_data).
    add_nearfield : bool
        If True, add a near-field region in front of the horn.
    resolution : int
        Number of radial points (default 30).

    Returns
    -------
    plotcoords_z : (resolution, n_points) ndarray
        Z-coordinates of the mesh grid.
    plotcoords_x : (resolution, n_points) ndarray
        Radial coordinates of the mesh grid.
    pmatx : (resolution, n_points) complex ndarray
        Complex pressure at each mesh point.
    """
    from mmm_toolbox.axi import make_km_axi

    n_modes = data["n_modes"]
    eigen_values = data["eigen_values"]

    U0 = np.zeros(n_modes, dtype=complex)
    U0[0] = data["St"]

    data["keep_zmatrix"] = True
    data["fvec"] = np.array([freq])
    data["nfreq"] = 1
    k_val = freq * 2.0 * np.pi / data["c"]
    data["k"] = np.array([k_val])

    data["Zrad"] = baffled_rad_zmatrix_direct_axi(
        data["k"], data["rho"], data["c"], data["Sm"],
        n_modes, eigen_values, use_hf_approx=True,
        n_quad=2000,
    )
    data = _calculate_matrices_single(data)

    stepped_coords = data["stepped_coords"]
    Nz = stepped_coords.shape[0]
    NP = int(np.ceil(Nz / 2 + 1))

    if add_nearfield:
        n_int = max(30, resolution + 5)
        data["n_integration_points"] = n_int
        dz_nf = data["c"] / (freq * 6.0)
        box_size = 2.0 * stepped_coords[-1, 1]
        Nfield = max(10, int(np.ceil(box_size / dz_nf)))
        mindist = 0.0
        NPcoords = NP + Nfield
        z_nf = np.linspace(mindist, mindist + box_size, Nfield)
        x_nf = np.linspace(0.0, box_size, resolution)
        Zm, X = np.meshgrid(z_nf, x_nf)
        field_points = np.column_stack([X.ravel(order="F"), Zm.ravel(order="F")])
    else:
        NPcoords = NP

    Pmat = np.zeros((n_modes, NP), dtype=complex)
    Pmatx = np.zeros((resolution, NPcoords), dtype=complex)
    plotcoords_z = np.zeros((resolution, NPcoords))
    plotcoords_x = np.zeros((resolution, NPcoords))

    step_indices = list(range(0, Nz, 2)) + [Nz - 1]
    plotcoords_z[:, :NP] = np.tile(
        stepped_coords[step_indices, 0],
        (resolution, 1),
    )

    if add_nearfield:
        plotcoords_x[:, NP:] = X
        plotcoords_z[:, NP:] = Zm + stepped_coords[-1, 0]

    ip = 0

    plotcoords_x[:, ip] = np.linspace(
        0.0, stepped_coords[0, 1], resolution,
    )
    phix = get_eigenfunctions_axi(
        stepped_coords[0, 1], plotcoords_x[:, ip], eigen_values, normalize=True,
    )
    Pmat[:, ip] = data["BigZ"][:, :, 0, 0] @ U0
    Pmatx[:, ip] = phix @ Pmat[:, ip]

    for iz in range(Nz - 1):
        R1 = stepped_coords[iz, 1]
        R2 = stepped_coords[iz + 1, 1]
        L = stepped_coords[iz + 1, 0] - stepped_coords[iz, 0]

        if L > 0.0:
            ip += 1
            krc = data["k"][0] * data["rho"] * data["c"]
            Z = data["BigZ"][:, :, iz, 0]
            kn = make_km_axi(data["k"][0], stepped_coords[iz, :], n_modes, eigen_values)

            D2 = 1j * np.sin(L * kn)
            Zc = krc / (data["S"][iz] * kn)
            invZc = (data["S"][iz] * kn) / krc
            E = np.diag(np.exp(-1j * L * kn))
            U0 = (-np.diag(D2 * invZc) @ (Z - np.diag(Zc)) + E) @ U0

            Pmat[:, ip] = data["BigZ"][:, :, iz + 1, 0] @ U0
            plotcoords_x[:, ip] = np.linspace(
                0.0, stepped_coords[iz + 1, 1], resolution,
            )
            phix = get_eigenfunctions_axi(
                stepped_coords[iz + 1, 1], plotcoords_x[:, ip],
                eigen_values, normalize=True,
            )
            Pmatx[:, ip] = phix @ Pmat[:, ip]
        else:
            F = data["big_f"][:, :, iz]
            if R1 > R2:
                U0 = np.linalg.solve(F.T, U0)
            else:
                U0 = F.T @ U0

    if add_nearfield:
        data_nf = _rayleigh_radiated_pressure(data, field_points)
        pRad = np.reshape(data_nf, (resolution, Nfield), order="F")
        Pmatx[:, NP:] = pRad

        ind = np.where(plotcoords_x[:, NP] <= stepped_coords[-1, 1])[0]
        coords_nf = plotcoords_x[ind, NP]
        phix_mouth = get_eigenfunctions_axi(
            stepped_coords[-1, 1], coords_nf, eigen_values, normalize=True,
        )
        p_mouth = phix_mouth @ Pmat[:, ip]
        Pmatx[ind, NP] = p_mouth

    return plotcoords_z, plotcoords_x, Pmatx


def _calculate_matrices_single(data: dict) -> dict:
    """Run calculate_matrices for a single-frequency data dict."""
    from mmm_toolbox.axi import make_km_axi

    n_modes = data["n_modes"]
    n_steps = data["stepped_coords"].shape[0]

    data["Umat"] = np.zeros((n_modes, n_modes, 1), dtype=complex)
    data["BigZ"] = np.zeros((n_modes, n_modes, n_steps, 1), dtype=complex)
    data["BigZ"][:, :, -1, :] = data["Zrad"]

    U = np.eye(n_modes, dtype=complex)
    Z = data["Zrad"][:, :, 0].copy()

    for iz in range(n_steps - 2, -1, -1):
        c1 = data["stepped_coords"][iz, :]
        c2 = data["stepped_coords"][iz + 1, :]
        L = c2[0] - c1[0]

        if L > 0.0:
            krc = data["k"][0] * data["rho"] * data["c"]
            kn = make_km_axi(data["k"][0], c1, n_modes, data["mode_info"])

            D2 = 1j * np.sin(L * kn)
            D3 = np.tan(L * kn)
            Zc = krc / (data["S"][iz] * kn)

            D2Zc = np.diag(Zc / D2)
            iD3Zc = np.diag(Zc / (1j * D3))
            Z = iD3Zc - D2Zc @ np.linalg.inv(Z + iD3Zc) @ D2Zc

            invZc = (data["S"][iz] * kn) / krc
            E = np.diag(np.exp(-1j * L * kn))
            U = U @ (-np.diag(D2 * invZc) @ (Z - np.diag(Zc)) + E)
        else:
            F = data["big_f"][:, :, iz]
            Ft = F.T
            Z = F @ Z @ Ft
            U = U @ Ft

        data["BigZ"][:, :, iz, 0] = Z

    data["Umat"][:, :, 0] = U
    data["Z00"] = data["BigZ"][0, 0, 0, 0]
    data["UmouthPw"] = data["Umat"][:, 0, 0]
    data["Umouth"] = data["UmouthPw"] * data["St"]

    return data
