"""Core simulation: initialization, F-matrix assembly, impedance propagation.

MATLAB originals:
  - MMM_init -> init_horn_data
  - MMM_makebigfmat -> make_big_fmat
  - MMM_calculateMatrices -> calculate_matrices
"""

import numpy as np

from mmm_toolbox.axi import _get_bessel_zeros, make_fmat_axi
from mmm_toolbox.geometry import make_steps


def make_big_fmat(
    n_modes: int, coords: np.ndarray, mode_info: np.ndarray, ffunc
) -> np.ndarray:
    """Assemble F scattering matrices at all duct discontinuities.

    Corresponds to MMM_makebigfmat.

    Returns (n_modes, n_modes, n_steps) array.
    """
    n_steps = coords.shape[0]
    big_f = np.zeros((n_modes, n_modes, n_steps))

    for iz in range(n_steps - 1):
        L = coords[iz + 1, 0] - coords[iz, 0]
        if L == 0.0:
            F = ffunc(n_modes, coords[iz, :], coords[iz + 1, :], mode_info)
            big_f[:, :, iz] = F

    return big_f


def init_horn_data(
    fvec: np.ndarray,
    n_modes: int,
    coords: np.ndarray,
    geometry: str,
    rho: float = 1.205,
    c: float = 344.0,
) -> dict:
    """Initialize the MMM data structure for a horn simulation.

    Corresponds to MMM_init.

    Returns a dictionary with keys:
      geometry, rho, c, fvec, nfreq, k, n_modes, keep_zmatrix,
      n_integration_points, raw_coords, stepped_coords, mode_index,
      mode_info, S, Sm, St, big_f, Zrad
    """
    if coords.size == 0:
        raise ValueError("Error: no horn coordinates.")
    if np.any(np.diff(coords[:, 0]) < 0):
        raise ValueError("Error: Z-coordinate is not monotonically increasing.")

    geometry = geometry.lower()
    nfreq = len(fvec)
    k = fvec * 2.0 * np.pi / c
    stepped_coords = make_steps(coords)

    S: np.ndarray
    eigen_values: np.ndarray

    if "axi" in geometry:
        bz = _get_bessel_zeros(n_modes)
        eigen_values = bz
        S = np.pi * stepped_coords[:, 1] ** 2
    elif "rect" in geometry:
        S = stepped_coords[:, 1] * stepped_coords[:, 2] * 4.0
        eigen_values = np.array([])
    else:
        raise ValueError(f"Unknown geometry type: {geometry}")

    data: dict = {
        "geometry": geometry,
        "rho": rho,
        "c": c,
        "fvec": fvec,
        "nfreq": nfreq,
        "k": k,
        "n_modes": n_modes,
        "keep_zmatrix": True,
        "n_integration_points": 20,
        "raw_coords": coords,
        "stepped_coords": stepped_coords,
        "mode_index": None,
        "eigen_values": eigen_values,
        "S": S,
        "mode_info": eigen_values,
        "Sm": S[-1],
        "St": S[0],
        "big_f": make_big_fmat(
            n_modes, stepped_coords, eigen_values, make_fmat_axi
        ),
        "Zrad": None,
    }

    return data

def calculate_matrices(data: dict, progress_report: bool = False) -> dict:
    """Propagate modal impedances and volume velocities mouth-to-throat.

    Corresponds to MMM_calculateMatrices.

    Modifies and returns data dict with added keys: BigZ, Umat, Z00,
    UmouthPw, Umouth.
    """
    from mmm_toolbox.axi import make_km_axi

    n_modes = data["n_modes"]
    nfreq = data["nfreq"]
    k_vec = data["k"]
    stepped_coords = data["stepped_coords"]
    n_steps = stepped_coords.shape[0]
    S = data["S"]
    big_f = data["big_f"]
    Zrad = data["Zrad"]
    mode_info = data["mode_info"]
    rho = data["rho"]
    c_sound = data["c"]
    keep_zmatrix = data["keep_zmatrix"]

    data["Umat"] = np.zeros((n_modes, n_modes, nfreq), dtype=complex)

    if keep_zmatrix:
        data["BigZ"] = np.zeros(
            (n_modes, n_modes, n_steps, nfreq), dtype=complex
        )
        data["BigZ"][:, :, -1, :] = Zrad

    for ik in range(nfreq):
        if progress_report:
            pct = ik / nfreq * 100.0
            print(
                f"Calculating k = {k_vec[ik]:.4f} ({pct:.1f}%)"
            )

        U = np.eye(n_modes, dtype=complex)
        Z = Zrad[:, :, ik].copy()

        # Propagate backward from mouth to throat
        for iz in range(n_steps - 2, -1, -1):
            c1 = stepped_coords[iz, :]
            stepped_coords[iz + 1, :]
            L = stepped_coords[iz + 1, 0] - stepped_coords[iz, 0]

            if L > 0.0:
                # Straight duct propagation (transmission line)
                krc = k_vec[ik] * rho * c_sound
                kn = make_km_axi(k_vec[ik], c1, n_modes, mode_info)

                D2 = 1j * np.sin(L * kn)
                D3 = np.tan(L * kn)
                Zc = krc / (S[iz] * kn)

                D2Zc = np.diag(Zc / D2)
                iD3Zc = np.diag(Zc / (1j * D3))

                Z = iD3Zc - D2Zc @ np.linalg.inv(Z + iD3Zc) @ D2Zc

                invZc = (S[iz] * kn) / krc
                E = np.diag(np.exp(-1j * L * kn))
                U = U @ (-np.diag(D2 * invZc) @ (Z - np.diag(Zc)) + E)

            else:
                # Discontinuity propagation (F-matrix)
                F = big_f[:, :, iz]
                Ft = F.T
                Z = F @ Z @ Ft
                U = U @ Ft

            if keep_zmatrix:
                data["BigZ"][:, :, iz, ik] = Z

        data["Umat"][:, :, ik] = U

    # Throat impedance (fundamental mode)
    data["Z00"] = np.squeeze(data["BigZ"][0, 0, 0, :])

    # Mouth volume velocity for throat plane wave excitation
    data["UmouthPw"] = np.squeeze(data["Umat"][:, 0, :])

    # Multiplication by throat area for default unit-throat-velocity
    data["Umouth"] = data["UmouthPw"] * data["St"]

    return data
