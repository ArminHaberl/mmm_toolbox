"""Test MMM_calculateMatrices — core impedance propagation algorithm."""

import numpy as np
from hornsim.core import calculate_matrices


def test_calculate(calculate_mat, init_mat, zrad_mat):
    """Full simulation: 8 modes, 200 frequencies, exponential horn."""
    expected_bigz = calculate_mat["BigZ"]
    expected_umat = calculate_mat["Umat"]
    expected_z00 = calculate_mat["Z00"].flatten()
    expected_umouthpw = calculate_mat["UmouthPw"]
    expected_umouth = calculate_mat["Umouth"]
    zrad = zrad_mat["Zrad"]

    data = {
        "n_modes": int(init_mat["nModes"].item()),
        "nfreq": int(init_mat["nfreq"].item()),
        "k": init_mat["kvec"].flatten(),
        "rho": init_mat["rho"].item(),
        "c": init_mat["c"].item(),
        "S": init_mat["S"].flatten(),
        "stepped_coords": init_mat["steppedCoords"],
        "mode_info": init_mat["eigenValues"].flatten(),
        "big_f": init_mat["bigF"],
        "Zrad": zrad,
        "keep_zmatrix": True,
    }

    data = calculate_matrices(data, progress_report=False)

    np.testing.assert_allclose(data["BigZ"], expected_bigz, atol=1e-10)
    np.testing.assert_allclose(data["Umat"], expected_umat, atol=1e-10)
    np.testing.assert_allclose(data["Z00"], expected_z00, atol=1e-10)
    np.testing.assert_allclose(data["UmouthPw"], expected_umouthpw, atol=1e-10)
    np.testing.assert_allclose(data["Umouth"], expected_umouth, atol=1e-10)
