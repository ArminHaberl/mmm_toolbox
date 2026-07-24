"""Horn geometry: contour generation and stepped discretization.

MATLAB originals:
  - MMM_1Dhorncoord -> horn_coord_1d
  - MMM_makesteps -> make_steps
"""

import numpy as np


def horn_coord_1d(
    htype: str,
    yth: float,
    ym: float,
    length: float,
    tn: float,
    dz: float,
    add_radius: bool = False,
    th_fta: float = 0.0,
    rad_r: float = 0.1,
    rad_fta: float = 80.0,
) -> np.ndarray:
    """Generate a 1D horn contour profile.

    Corresponds to MMM_1Dhorncoord.

    Returns (N, 2) array: columns are (z, radius).
    """
    raise NotImplementedError


def make_steps(coords: np.ndarray) -> np.ndarray:
    """Convert smooth horn profile to stepped duct approximation.

    Corresponds to MMM_makesteps.

    Parameters
    ----------
    coords : (N, D) array
        Smooth horn coordinates. Column 0 is z, columns 1+ are radii.

    Returns
    -------
    newcoords : (M, D) array
        Stepped coordinates.
    """
    raise NotImplementedError
