"""Visualization and post-processing: directivity index, polar maps.

MATLAB originals:
  - MMM_ASgetDI -> get_di_axi
"""

import numpy as np


def get_di_axi(data: dict, angles: np.ndarray) -> np.ndarray:
    """Compute directivity index from radiated pressure field.

    Corresponds to MMM_ASgetDI.

    Modifies data dict in-place: sets data['DI'].
    Returns data for convenience.
    """
    raise NotImplementedError
