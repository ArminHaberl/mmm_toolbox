"""End-to-end integration test: full horn simulation pipeline.

Replicates AxiHorndemo1 workflow: exponential horn, 8 modes, 100–15000 Hz.
Validates Z00 (throat impedance) — the primary engineering output.
"""

from pathlib import Path

import numpy as np

from hornsim.core import calculate_matrices, init_horn_data
from hornsim.geometry import horn_coord_1d
from hornsim.plotting import get_di_axi
from hornsim.radiation import baffled_rad_zmatrix_axi, radiated_pressure_axi

DATA_DIR = Path(__file__).parent.parent / "hornsim" / "data"


def test_full_pipeline(
    horncoords_mat, init_mat, zrad_mat, calculate_mat, prad_mat, di_mat,
):
    """Run the full AxiHorndemo1 workflow and compare against MATLAB."""

    # --- Build inputs from horncoords data ---
    htype = str(horncoords_mat["HornType"][0])
    rth = horncoords_mat["rth"].item()
    rm = horncoords_mat["rm"].item()
    L = horncoords_mat["Lh"].item()
    T = horncoords_mat["T"].item()
    dz = horncoords_mat["dz"].item()

    # --- 1. Horn contour ---
    horncoords = horn_coord_1d(htype, rth, rm, L, T, dz)
    np.testing.assert_allclose(horncoords, horncoords_mat["horncoords"], atol=1e-12)

    # --- 2. Init ---
    freq = init_mat["freq"].flatten()
    n_modes = int(init_mat["nModes"].item())
    rho = init_mat["rho"].item()
    c = init_mat["c"].item()

    data = init_horn_data(freq, n_modes, horncoords, "axi", rho, c)
    expected_z00 = calculate_mat["Z00"].flatten()

    # --- 3. Radiation impedance ---
    Zrad = baffled_rad_zmatrix_axi(data["k"], rho, c, data["Sm"], n_modes, str(DATA_DIR / "ZradAS32.mat"))
    data["Zrad"] = Zrad
    np.testing.assert_allclose(Zrad, zrad_mat["Zrad"], atol=1e-10)

    # --- 4. Calculate matrices (core simulation) ---
    data = calculate_matrices(data, progress_report=False)
    np.testing.assert_allclose(data["Z00"], expected_z00, atol=1e-10)

    # --- 5. Radiated pressure ---
    field_points = prad_mat["fieldPoints"]
    data = radiated_pressure_axi(data, field_points, use_farfield_approx=True)
    np.testing.assert_allclose(data["pRad"], prad_mat["pRad"], atol=1e-8)

    # --- 6. Directivity index ---
    angles = di_mat["Angext"].flatten()
    data = get_di_axi(data, angles)
    np.testing.assert_allclose(data["DI"], di_mat["DI"].flatten(), atol=1e-10)
