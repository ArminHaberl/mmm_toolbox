"""Axisymmetric mode-matching functions.

MATLAB originals:
  - MMM_ASmakefmat -> make_fmat_axi
  - MMM_ASmakekm -> make_km_axi
  - MMM_ASgeteigenfunctions -> get_eigenfunctions_axi
"""

import numpy as np
from scipy.special import j0, j1, jn_zeros


def _get_bessel_zeros(n: int) -> np.ndarray:
    """Return the first *n* zeros of J₁, with a leading zero.

    The leading zero corresponds to the fundamental plane-wave mode
    (eigenvalue 0).  The remaining *n*‑1 values are the positive zeros
    α₁, α₂, … used by the axisymmetric mode-matching formulation.
    """
    return np.concatenate([[0.0], jn_zeros(1, n - 1)])


def make_fmat_axi(
    n_modes: int, coord1: np.ndarray, coord2: np.ndarray, bz: np.ndarray
) -> np.ndarray:
    """Compute the F coupling matrix for an axisymmetric duct discontinuity.

    Corresponds to MMM_ASmakefmat.

    Parameters
    ----------
    n_modes : int
        Number of modes.
    coord1 : (2,) array
        (z, radius) of upstream duct.
    coord2 : (2,) array
        (z, radius) of downstream duct.
    bz : (n_modes,) array
        Zeros of Bessel function J1 (eigenvalues).

    Returns
    -------
    F : (n_modes, n_modes) ndarray
        Mode coupling (scattering) matrix.
    """
    R1 = coord1[1]
    R2 = coord2[1]
    beta = R1 / R2

    if beta > 1.0:
        beta = 1.0 / beta
    elif beta == 1.0:
        return np.eye(n_modes)

    gamma_n = bz[:n_modes, np.newaxis]  # (N, 1) — gamma_n[i,j] = bz[i]
    gamma_m = bz[np.newaxis, :n_modes]  # (1, N) — gamma_m[i,j] = bz[j]

    Fm = 2.0 * beta * gamma_m * j1(beta * gamma_m) / j0(gamma_m)
    F = Fm / (beta**2 * gamma_m**2 - gamma_n**2)

    F[0, 0] = 1.0
    return F


def make_km_axi(
    k: float, coord: np.ndarray, n_modes: int, bz: np.ndarray
) -> np.ndarray:
    """Compute modal wavenumbers for an axisymmetric duct.

    Corresponds to MMM_ASmakekm.

    Parameters
    ----------
    k : float
        Free-space wavenumber.
    coord : (2,) array
        (z, radius) of the duct.
    n_modes : int
        Number of modes.
    bz : (n_modes,) array
        Zeros of Bessel function J1.

    Returns
    -------
    km : (n_modes,) ndarray
        Modal wavenumbers (complex for evanescent modes).
    """
    R = coord[1]
    gmR = bz[:n_modes] / R
    delta = k**2 - gmR**2
    km = np.conj(np.sqrt(delta + 0j))
    return km


def get_eigenfunctions_axi(
    radius: float,
    rcoords: np.ndarray,
    eigen_values: np.ndarray,
    normalize: bool = True,
) -> np.ndarray:
    """Compute axisymmetric eigenfunctions J0(alpha_n * r / R).

    Corresponds to MMM_ASgeteigenfunctions.

    Parameters
    ----------
    radius : float
        Duct radius R.
    rcoords : (Nr,) array
        Radial coordinates at which to evaluate.
    eigen_values : (n_modes,) array
        Zeros of J1 (eigenvalues).
    normalize : bool
        If True, normalize by J0(gamma_n).

    Returns
    -------
    phi : (Nr, n_modes) ndarray
        Eigenfunction values.
    """
    alpha = eigen_values / radius  # (n_modes,)
    arg = alpha[:, np.newaxis] * rcoords[np.newaxis, :]  # (n_modes, Nr)
    phi = j0(arg).T  # (Nr, n_modes)

    if normalize:
        norm = j0(eigen_values)  # (n_modes,)
        phi = phi / norm[np.newaxis, :]

    return phi
