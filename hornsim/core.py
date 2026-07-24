"""Core simulation: initialization, F-matrix assembly, impedance propagation.

MATLAB originals:
  - MMM_init -> init_horn_data
  - MMM_makebigfmat -> make_big_fmat
  - MMM_calculateMatrices -> calculate_matrices
"""

import numpy as np


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
    raise NotImplementedError


def make_big_fmat(
    n_modes: int, coords: np.ndarray, mode_info: np.ndarray, ffunc
) -> np.ndarray:
    """Assemble F scattering matrices at all duct discontinuities.

    Corresponds to MMM_makebigfmat.

    Returns (n_modes, n_modes, n_steps) array.
    """
    raise NotImplementedError


def calculate_matrices(data: dict, progress_report: bool = False) -> dict:
    """Propagate modal impedances and volume velocities mouth-to-throat.

    Corresponds to MMM_calculateMatrices.

    Modifies and returns data dict with added keys: BigZ, Umat, Z00,
    UmouthPw, Umouth.
    """
    raise NotImplementedError
