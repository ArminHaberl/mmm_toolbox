"""Radiation impedance and far-field pressure calculation.

MATLAB originals:
  - MMM_ASbaffledradzmatrixIntp -> baffled_rad_zmatrix_axi
  - MMM_ASradiatedPressure -> radiated_pressure_axi
"""

import numpy as np


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
    raise NotImplementedError


def radiated_pressure_axi(
    data: dict,
    field_points: np.ndarray,
    use_farfield_approx: bool = True,
) -> np.ndarray:
    """Calculate radiated pressure at given field points.

    Corresponds to MMM_ASradiatedPressure.

    Modifies data dict in-place: sets data['pRad'].
    Returns data for convenience.
    """
    raise NotImplementedError
