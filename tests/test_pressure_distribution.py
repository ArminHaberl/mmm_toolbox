"""Test MMM_ASpressureDistribution — internal and near-field pressure field."""

import numpy as np

from mmm_toolbox.geometry import horn_coord_1d
from mmm_toolbox.radiation import pressure_distribution_axi


def test_pressure_distribution_internal(
    pressure_dist_mat, init_mat, bessel_zeros,
):
    """Internal horn pressure field (no nearfield) — tractrix, 1500 Hz, 24 modes."""
    freq = pressure_dist_mat["freq"].item()
    n_modes = int(pressure_dist_mat["N"].item())
    rho = init_mat["rho"].item()
    c = init_mat["c"].item()

    sth = 10e-4
    sm = 1500e-4
    Lh = 49e-2
    dz = Lh / 250
    rth = np.sqrt(sth / np.pi)
    rm = np.sqrt(sm / np.pi)
    horncoords = horn_coord_1d("tractrix", rth, rm, Lh, 1.0, dz)

    from mmm_toolbox.core import init_horn_data
    data = init_horn_data(np.array([freq]), n_modes, horncoords, "axi", rho, c)

    plotcoordsz, plotcoordsx, Pmatx = pressure_distribution_axi(
        freq, data, add_nearfield=False, resolution=30,
    )

    expected_z = pressure_dist_mat["plotcoordsz"]
    expected_x = pressure_dist_mat["plotcoordsx"]
    expected_P = pressure_dist_mat["Pmatx"]

    np.testing.assert_allclose(plotcoordsz, expected_z, atol=1e-12)
    np.testing.assert_allclose(plotcoordsx, expected_x, atol=1e-12)
    np.testing.assert_allclose(Pmatx, expected_P, atol=0.02)


def test_pressure_distribution_nearfield(
    pressure_dist_nf_mat, init_mat, bessel_zeros,
):
    """Internal + near-field horn pressure — tractrix, 1500 Hz, 24 modes."""
    n_modes = 24
    rho = init_mat["rho"].item()
    c = init_mat["c"].item()

    sth = 10e-4
    sm = 1500e-4
    Lh = 49e-2
    dz = Lh / 250
    rth = np.sqrt(sth / np.pi)
    rm = np.sqrt(sm / np.pi)
    horncoords = horn_coord_1d("tractrix", rth, rm, Lh, 1.0, dz)

    from mmm_toolbox.core import init_horn_data
    data = init_horn_data(np.array([1500.0]), n_modes, horncoords, "axi", rho, c)

    plotcoordsz, plotcoordsx, Pmatx = pressure_distribution_axi(
        1500.0, data, add_nearfield=True, resolution=30,
    )

    expected_z = pressure_dist_nf_mat["plotcoordsz_nf"]
    expected_x = pressure_dist_nf_mat["plotcoordsx_nf"]
    expected_P = pressure_dist_nf_mat["Pmatx_nf"]

    np.testing.assert_allclose(plotcoordsz, expected_z, atol=1e-12)
    np.testing.assert_allclose(plotcoordsx, expected_x, atol=1e-12)
    np.testing.assert_allclose(Pmatx, expected_P, atol=0.02)
