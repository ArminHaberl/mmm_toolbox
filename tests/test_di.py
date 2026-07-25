"""Test MMM_ASgetDI — directivity index calculation."""

import numpy as np

from mmm_toolbox.plotting import get_di_axi


def test_di(di_mat, prad_mat):
    """Directivity index from 181 angles, 200 frequencies."""
    expected = di_mat["DI"].flatten()
    angles = di_mat["Angext"].flatten()
    prad = prad_mat["pRad"]

    data = {"pRad": prad, "nfreq": prad.shape[1]}
    data = get_di_axi(data, angles)

    np.testing.assert_allclose(data["DI"], expected, atol=1e-10)


def test_di_gerzon(di_gerzon_mat, init_mat, horncoords_mat):
    """Directivity index from 24 angles — Gerzon weighting path."""
    from pathlib import Path

    from mmm_toolbox.core import calculate_matrices, init_horn_data
    from mmm_toolbox.radiation import baffled_rad_zmatrix_axi, radiated_pressure_axi

    DATA_DIR = Path(__file__).parent.parent / "mmm_toolbox" / "data"

    expected = di_gerzon_mat["DI_gerzon"].flatten()
    angles = di_gerzon_mat["Angext_coarse"].flatten()

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

    Rext = 3.0
    field_points = np.column_stack([
        Rext * np.sin(np.deg2rad(angles)),
        Rext * np.cos(np.deg2rad(angles)),
    ])
    data = radiated_pressure_axi(data, field_points, use_farfield_approx=True)
    data = get_di_axi(data, angles)

    np.testing.assert_allclose(data["DI"], expected, atol=1e-10)
