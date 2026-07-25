#!/usr/bin/env python3
"""AxiHorndemo1: full axisymmetric horn loudspeaker simulation.

Python port of matlab/AxiHorndemo1.m from the MMM Toolbox by Bjørn Kolbrek.

Calculates the performance of an exponential horn excited by a plane wave
at the throat.  The workflow:

  1. Generate horn contour → stepped duct approximation
  2. Initialise the data structure (modes, frequencies, materials)
  3. Modal radiation impedance at the mouth (precomputed table, interpolated)
  4. Propagate impedances & velocities mouth-to-throat through the duct
  5. Far-field radiated pressure at field points (0°–90°, 3 m)
  6. Visualisation: horn profile, throat impedance, SPL curves, polar map

Requires matplotlib (``pip install matplotlib``) in addition to the
standard mmm_toolbox dependencies.

----------------------------------------------------------------------
  This file is part of the Mode Matching Method (MMM) Toolbox by
  Bjørn Kolbrek.  Copyright (C) 2012-2025 by Bjørn Kolbrek.

  The MMM Toolbox is free software: you can redistribute it and/or
  modify it under the terms of the GNU General Public License as
  published by the Free Software Foundation, either version 2 of the
  License, or (at your option) any later version.

  The MMM Toolbox is distributed in the hope that it will be useful,
  but WITHOUT ANY WARRANTY; without even the implied warranty of
  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
  General Public License for more details.

  You should have received a copy of the GNU General Public License
  along with the MMM Toolbox.  If not, see <http://www.gnu.org/licenses/>.
----------------------------------------------------------------------
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from mmm_toolbox.core import calculate_matrices, init_horn_data
from mmm_toolbox.geometry import horn_coord_1d
from mmm_toolbox.plotting import (
    get_di_axi,
    plot_directivity_index,
    plot_horn_profile,
    plot_polar_map,
    plot_spl_vs_frequency,
    plot_throat_impedance,
)
from mmm_toolbox.radiation import baffled_rad_zmatrix_axi, radiated_pressure_axi

# -----------------------------------------------------------------------
#  Horn parameters
# -----------------------------------------------------------------------
Sth = 10e-4  # throat area  [m²]
Sm = 500e-4  # mouth area   [m²]
Lh = 30e-2  # horn length  [m]
dz = Lh / 250  # segment length
Tn = 1  # horn parameter (hypex / Bessel)
HornType = "exponential"
AddRadius = False
# RadR = 0.1        # mouth radius (unused when AddRadius=False)
# RFta = 80         # end tangent angle (unused when AddRadius=False)

# -----------------------------------------------------------------------
#  Field-point parameters
# -----------------------------------------------------------------------
Rext = 3.0  # field-point distance  [m]
Angext = np.linspace(0.0, 90.0, 181)  # field-point angles  [deg]
use_farfield_approx = True

# -----------------------------------------------------------------------
#  Simulation parameters
# -----------------------------------------------------------------------
fmin, fmax, Nf = 100.0, 15000.0, 200
N = 8  # maximum number of modes
c, rho = 344.0, 1.205  # sound speed [m/s], air density [kg/m³]
freq = np.logspace(np.log10(fmin), np.log10(fmax), Nf)

# -----------------------------------------------------------------------
#  1.  Horn contour
# -----------------------------------------------------------------------
rth = np.sqrt(Sth / np.pi)
rm = np.sqrt(Sm / np.pi)
horncoords = horn_coord_1d(HornType, rth, rm, Lh, Tn, dz)
#   (AddRadius radiused flare is not yet ported; skipped.)

# -----------------------------------------------------------------------
#  2.  Initialisation
# -----------------------------------------------------------------------
data = init_horn_data(freq, N, horncoords, "axi", rho, c)

# -----------------------------------------------------------------------
#  3.  Radiation impedance at the mouth
# -----------------------------------------------------------------------
print("Calculating radiation impedance ...")
data["Zrad"] = baffled_rad_zmatrix_axi(
    data["k"],
    rho,
    c,
    data["Sm"],
    data["n_modes"],
)

# -----------------------------------------------------------------------
#  4.  Impedance / velocity propagation
# -----------------------------------------------------------------------
print("Calculating horn matrices ...")
data = calculate_matrices(data, progress_report=False)

# -----------------------------------------------------------------------
#  5.  Radiated pressure at field points
# -----------------------------------------------------------------------
field_points = np.column_stack(
    (Rext * np.sin(np.deg2rad(Angext)), Rext * np.cos(np.deg2rad(Angext)))
)
print("Calculating field point pressure ...")
data = radiated_pressure_axi(
    data, field_points, use_farfield_approx=use_farfield_approx
)

# -----------------------------------------------------------------------
#  6.  Visualisation  (requires matplotlib)
# -----------------------------------------------------------------------
import matplotlib.pyplot as plt

fig1, _ = plot_horn_profile(data)
fig2, _ = plot_throat_impedance(data)
fig3, _ = plot_spl_vs_frequency(data, Angext)

data = get_di_axi(data, Angext)

fig4 = plt.figure(figsize=(9, 9))
_, _ = plot_polar_map(data, Angext, ax=fig4.add_subplot(2, 1, 1))
_, _ = plot_directivity_index(data, ax=fig4.add_subplot(2, 1, 2))
fig4.tight_layout()

out_dir = Path.cwd() / "figures"
out_dir.mkdir(exist_ok=True)
for i, fig in enumerate((fig1, fig2, fig3, fig4), start=1):
    fig.savefig(out_dir / f"axi_horn_demo1_fig{i}.png", dpi=150)
print(f"Plots saved to {out_dir.resolve()}/")
