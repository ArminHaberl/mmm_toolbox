"""Test MMM_ASradiatedPressure — Rayleigh integral near-field pressure."""

from pathlib import Path

import numpy as np

from mmm_toolbox.core import calculate_matrices, init_horn_data
from mmm_toolbox.radiation import baffled_rad_zmatrix_axi, radiated_pressure_axi

DATA_DIR = Path(__file__).parent.parent / "test_data"


def test_rayleigh_matlab(prad_rayleigh_mat, init_mat, horncoords_mat):
    """Rayleigh integral should match MATLAB output on 7 angles."""
    freq = init_mat["freq"].flatten()
    n_modes = int(init_mat["nModes"].item())
    horncoords = horncoords_mat["horncoords"]
    rho = init_mat["rho"].item()
    c = init_mat["c"].item()

    data = init_horn_data(freq, n_modes, horncoords, "axi", rho, c)
    data["Zrad"] = baffled_rad_zmatrix_axi(
        data["k"], rho, c, data["Sm"], n_modes, str(DATA_DIR / "ZradAS32.mat"),
    )
    data = calculate_matrices(data, progress_report=False)

    field_points = prad_rayleigh_mat["fieldPoints_nf"]
    expected = prad_rayleigh_mat["pRad_rayleigh"]

    data = radiated_pressure_axi(data, field_points, use_farfield_approx=False)

    np.testing.assert_allclose(data["pRad"], expected, atol=1e-8)
