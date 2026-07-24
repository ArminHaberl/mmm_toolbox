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

from hornsim.axi import get_eigenfunctions_axi, make_km_axi
from hornsim.core import calculate_matrices, init_horn_data
from hornsim.geometry import horn_coord_1d
from hornsim.radiation import baffled_rad_zmatrix_axi, radiated_pressure_axi

DATA_DIR = Path(__file__).resolve().parent.parent / "hornsim" / "data"

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
_zrad_path = str(DATA_DIR / "ZradAS32.mat")
data["Zrad"] = baffled_rad_zmatrix_axi(
    data["k"], rho, c, data["Sm"], data["n_modes"], _zrad_path,
)

# -----------------------------------------------------------------------
#  4.  Impedance / velocity propagation
# -----------------------------------------------------------------------
data = calculate_matrices(data, progress_report=False)

# -----------------------------------------------------------------------
#  5.  Internal pressure distribution
# -----------------------------------------------------------------------
n_modes = data["n_modes"]
stepped_coords = data["stepped_coords"]
Nz = stepped_coords.shape[0]
eigen_values = data["eigen_values"]
resolution = 30

# Throat plane-wave excitation
U0 = np.zeros(n_modes, dtype=complex)
U0[0] = data["St"]

# Axial positions at which to sample: every second step + the last
sample_indices = list(range(0, Nz, 2)) + [Nz - 1]
NP = len(sample_indices)

if add_nearfield:
    data["n_integration_points"] = max(30, resolution + 5)
    box_size = 2.0 * stepped_coords[-1, 1]
    Nfield = max(10, int(np.ceil(box_size / (c / (freq[0] * 6)))))
    z_nf = np.linspace(0.0, box_size, Nfield)
    x_nf = np.linspace(0.0, box_size, resolution)
    Zm, X = np.meshgrid(z_nf, x_nf)
    field_points = np.column_stack((X.ravel(), Zm.ravel()))
    NPcoords = NP + Nfield
else:
    NPcoords = NP

Pmat = np.zeros((n_modes, NP), dtype=complex)
Pmatx = np.zeros((resolution, NPcoords), dtype=complex)
plotcoordsz = np.zeros((resolution, NPcoords))
plotcoordsx = np.zeros((resolution, NPcoords))

# Axial z-coordinates for the sampling planes
zst = stepped_coords[:, 0]
plotcoordsz[:, :NP] = np.tile(zst[sample_indices], (resolution, 1))

if add_nearfield:
    plotcoordsx[:, NP:] = X
    plotcoordsz[:, NP:] = Zm + stepped_coords[-1, 0]

# --- Internal pressure along the horn ---
ip = 0
ik = 0  # single frequency

# At the throat
plotcoordsx[:, ip] = np.linspace(0.0, stepped_coords[0, 1], resolution)
phi = get_eigenfunctions_axi(stepped_coords[0, 1], plotcoordsx[:, ip], eigen_values, True)
Pmat[:, ip] = data["BigZ"][:, :, 0, ik] @ U0
Pmatx[:, ip] = phi @ Pmat[:, ip]

U = U0.copy()
for iz in range(Nz - 1):
    R1 = stepped_coords[iz, 1]
    R2 = stepped_coords[iz + 1, 1]
    L = stepped_coords[iz + 1, 0] - stepped_coords[iz, 0]

    if L > 0.0:
        ip += 1
        Z = data["BigZ"][:, :, iz, ik]
        dup = stepped_coords[iz, :]
        kn = make_km_axi(data["k"][ik], dup, n_modes, eigen_values)
        krc = data["k"][ik] * rho * c

        Zc = krc / (data["S"][iz] * kn)
        D2 = 1j * np.sin(L * kn)
        D2Zc = np.diag(D2 / Zc)
        E = np.diag(np.exp(-1j * L * kn))

        U = (E - D2Zc @ (Z - np.diag(Zc))) @ U

        Pmat[:, ip] = data["BigZ"][:, :, iz + 1, ik] @ U
        plotcoordsx[:, ip] = np.linspace(0.0, stepped_coords[iz + 1, 1], resolution)
        phi = get_eigenfunctions_axi(stepped_coords[iz + 1, 1], plotcoordsx[:, ip], eigen_values, True)
        Pmatx[:, ip] = phi @ Pmat[:, ip]
    else:
        F = data["big_f"][:, :, iz]
        if R1 > R2:
            U = np.linalg.solve(F.T, U)
        else:
            U = F.T @ U

# --- Near-field radiated pressure in front of the horn ---
if add_nearfield:
    data["Umouth"] = data["Umouth"].reshape(n_modes, -1)
    data_nf = radiated_pressure_axi(
        data, field_points, use_farfield_approx=True,
    )
    p_rad = data_nf["pRad"].reshape(resolution, Nfield)
    Pmatx[:, NP:] = p_rad

    # Replace first column of nearfield with eigenfunction-projected values
    idx = np.where(plotcoordsx[:, NP] <= stepped_coords[-1, 1])[0]
    if len(idx) > 0:
        coords_mouth = plotcoordsx[idx, NP]
        phi_mouth = get_eigenfunctions_axi(stepped_coords[-1, 1], coords_mouth, eigen_values, True)
        Pmatx[idx, NP] = phi_mouth @ Pmat[:, ip]

# -----------------------------------------------------------------------
#  6.  Visualisation  (requires matplotlib)
# -----------------------------------------------------------------------
import matplotlib.pyplot as plt

# Figure 1: Horn profile
fig1, ax1 = plt.subplots(figsize=(8, 5))
rc = data["raw_coords"]
sc = stepped_coords
ym = np.max(rc[:, 1])
ax1.plot(sc[:, 0], sc[:, 1], "b", sc[:, 0], -sc[:, 1], "b",
        rc[:, 0], rc[:, 1], "k", rc[:, 0], -rc[:, 1], "k")
ax1.set_ylim(-ym * 1.1, ym * 1.1)
ax1.set_aspect("equal")
ax1.set_xlabel("z axis [m]")
ax1.set_ylabel("Radius [m]")
ax1.set_title("Horn profile")
fig1.tight_layout()

# Figure 2: Internal sound field
fig2, ax2 = plt.subplots(figsize=(10, 6))
SPL = 94.0 + 20.0 * np.log10(np.abs(Pmatx))
cf = ax2.contourf(plotcoordsz.T, plotcoordsx.T, SPL.T, 25, cmap="viridis")
fig2.colorbar(cf, ax=ax2, label="dB SPL")
ax2.set_aspect("equal")

# Overlay horn outline
coords_double = np.vstack((rc, rc[-1, :]))
coords_double[-1, 1] = 0.0
ax2.plot(coords_double[:, 0], coords_double[:, 1], "k", linewidth=1.5)
ax2.plot(coords_double[:, 0], -coords_double[:, 1], "k", linewidth=1.5)

ax2.set_title(f"Sound field at {freq[0]:.0f} Hz, using {N} modes")
ax2.set_xlabel("z [m]")
ax2.set_ylabel("Radius [m]")
fig2.tight_layout()

out_dir = Path.cwd() / "figures"
out_dir.mkdir(exist_ok=True)
fig1.savefig(out_dir / "axi_horn_demo2_fig1.png", dpi=150)
fig2.savefig(out_dir / "axi_horn_demo2_fig2.png", dpi=150)
print(f"Plots saved to {out_dir.resolve()}/")
