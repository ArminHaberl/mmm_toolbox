#!/usr/bin/env python3
"""AxiHorndemo2: internal pressure field at a single frequency.

Python port of matlab/AxiHorndemo2.m from the MMM Toolbox by Bjørn Kolbrek.

Calculates and visualises the pressure field inside and in front of
a horn at a single frequency.  Adjusting the number of modes lets
you observe convergence behaviour — too few modes produce wiggly
equi-pressure contours.

The workflow:

  1.  Generate horn contour → stepped duct approximation
  2.  Initialise the data structure (24 modes, single frequency)
  3.  Modal radiation impedance at the mouth
  4.  Propagate impedances & velocities through the duct
  5.  Compute pressure distribution on a mesh inside the horn
  6.  (optional) Near-field radiated pressure in front of the horn
  7.  Visualisation: horn profile + contour plot of SPL

Requires matplotlib (``pip install matplotlib``).

----------------------------------------------------------------------
  This file is part of the Mode Matching Method (MMM) Toolbox by
  Bjørn Kolbrek.  Copyright (C) 2012-2025 by Bjørn Kolbrek.

  The MMM Toolbox is free software: you can redistribute it and/or
  modify it under the terms of the GNU General Public License as
  published by the Free Software Foundation, either version 2 of the
  License, or (at your option) any later version.
----------------------------------------------------------------------
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from mmm_toolbox.core import calculate_matrices, init_horn_data
from mmm_toolbox.geometry import horn_coord_1d
from mmm_toolbox.plotting import plot_horn_profile, plot_sound_field
from mmm_toolbox.radiation import baffled_rad_zmatrix_axi

# -----------------------------------------------------------------------
#  Horn parameters
# -----------------------------------------------------------------------
Sth = 10e-4    # throat area  [m²]
Sm = 1500e-4   # mouth area   [m²]
Lh = 49e-2     # horn length  [m]
dz = Lh / 250  # segment length
Tn = 1.0
HornType = "tractrix"
AddRadius = False

# -----------------------------------------------------------------------
#  Simulation parameters
# -----------------------------------------------------------------------
freq = np.array([1500.0])   # single frequency  [Hz]
N = 24                       # max number of modes
c = 344.0
rho = 1.205
add_nearfield = True

# -----------------------------------------------------------------------
#  1.  Horn contour
# -----------------------------------------------------------------------
rth = np.sqrt(Sth / np.pi)
rm = np.sqrt(Sm / np.pi)
horncoords = horn_coord_1d(HornType, rth, rm, Lh, Tn, dz)

# -----------------------------------------------------------------------
#  2.  Initialisation
# -----------------------------------------------------------------------
data = init_horn_data(freq, N, horncoords, "axi", rho, c)

# -----------------------------------------------------------------------
#  3.  Radiation impedance at the mouth
# -----------------------------------------------------------------------
data["Zrad"] = baffled_rad_zmatrix_axi(
    data["k"], rho, c, data["Sm"], data["n_modes"],
)

# -----------------------------------------------------------------------
#  4.  Impedance / velocity propagation
# -----------------------------------------------------------------------
data = calculate_matrices(data, progress_report=False)

# -----------------------------------------------------------------------
#  5.  Internal pressure distribution + Visualisation
# -----------------------------------------------------------------------
fig1, _ = plot_horn_profile(data)
fig2, _ = plot_sound_field(
    data, freq=float(freq[0]), add_nearfield=add_nearfield, resolution=30,
)

out_dir = Path.cwd() / "figures"
out_dir.mkdir(exist_ok=True)
fig1.savefig(out_dir / "axi_horn_demo2_fig1.png", dpi=150)
fig2.savefig(out_dir / "axi_horn_demo2_fig2.png", dpi=150)
print(f"Plots saved to {out_dir.resolve()}/")
