"""Axisymmetric mode-matching functions.

MATLAB originals:
  - MMM_ASmakefmat -> make_fmat_axi
  - MMM_ASmakekm -> make_km_axi
  - MMM_ASgeteigenfunctions -> get_eigenfunctions_axi
"""

import numpy as np


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
    raise NotImplementedError


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
    raise NotImplementedError


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
    raise NotImplementedError
