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
standard hornsim dependencies.

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

from hornsim.core import calculate_matrices, init_horn_data
from hornsim.geometry import horn_coord_1d
from hornsim.plotting import get_di_axi
from hornsim.radiation import baffled_rad_zmatrix_axi, radiated_pressure_axi

DATA_DIR = Path(__file__).resolve().parent.parent / "hornsim" / "data"

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
_zrad_path = str(DATA_DIR / "ZradAS32.mat")
data["Zrad"] = baffled_rad_zmatrix_axi(
    data["k"],
    rho,
    c,
    data["Sm"],
    data["n_modes"],
    _zrad_path,
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
import matplotlib.pyplot as plt  # noqa: E402

# -- Figure 1: Horn profile -------------------------------------------
fig1, ax1 = plt.subplots(figsize=(8, 5))
rc = data["raw_coords"]
sc = data["stepped_coords"]
ym = np.max(rc[:, 1])

ax1.plot(
    sc[:, 0],
    sc[:, 1],
    "b",
    sc[:, 0],
    -sc[:, 1],
    "b",
    rc[:, 0],
    rc[:, 1],
    "k",
    rc[:, 0],
    -rc[:, 1],
    "k",
)
ax1.set_ylim(-ym * 1.1, ym * 1.1)
ax1.set_aspect("equal")
ax1.set_xlabel("z axis [m]")
ax1.set_ylabel("Radius [m]")
ax1.set_title("Horn profile")
fig1.tight_layout()

# -- Figure 2: Normalised throat impedance ----------------------------
fig2, ax2 = plt.subplots(figsize=(8, 5))
Z00n = data["St"] / (data["rho"] * data["c"]) * data["Z00"]

ax2.semilogx(data["fvec"], np.real(Z00n), "k", label="Real")
ax2.semilogx(data["fvec"], np.imag(Z00n), "r", label="Imag")
ax2.set_xlim(data["fvec"][0], data["fvec"][-1])
ax2.set_xlabel("Hz")
ax2.set_ylabel("Normalized acoustic Z")
ax2.set_title("Horn throat impedance")
ax2.legend()
ax2.grid(True)
fig2.tight_layout()

# -- Figure 3: Field-point SPL vs frequency -----------------------
fig3, ax3 = plt.subplots(figsize=(9, 6))
ia = np.where(np.mod(Angext, 10) == 0)[0]
SPL = 94.0 + 20.0 * np.log10(np.abs(data["pRad"]))
for idx in ia:
    ax3.semilogx(data["fvec"], SPL[idx, :], label=f"{Angext[idx]:.0f}°")
ax3.set_xlim(data["fvec"][0], data["fvec"][-1])
ax3.set_xlabel("Hz")
ax3.set_ylabel("dB SPL")
ax3.set_title("Field point pressures")
ax3.legend(title="Angle", ncol=2, fontsize="small")
ax3.grid(True)
fig3.tight_layout()

# -- Figure 4: Polar map + Directivity index --------------------------
data = get_di_axi(data, Angext)
Lp = 20.0 * np.log10(np.abs(data["pRad"]))
Lp_norm = Lp - Lp[0, :]  # normalise to on-axis

fig4 = plt.figure(figsize=(9, 9))

ax4a = fig4.add_subplot(2, 1, 1)
cf = ax4a.contourf(data["fvec"], Angext, Lp_norm, 15, cmap="viridis")
ax4a.set_xscale("log")
ax4a.set_xlim(data["fvec"][0], data["fvec"][-1])
ax4a.set_xlabel("Hz")
ax4a.set_ylabel("Degrees")
ax4a.set_title("Polar map (normalized)")
fig4.colorbar(cf, ax=ax4a, label="dB")

ax4b = fig4.add_subplot(2, 1, 2)
ax4b.semilogx(data["fvec"], data["DI"], "k")
ax4b.set_xlim(data["fvec"][0], data["fvec"][-1])
ax4b.set_xlabel("Hz")
ax4b.set_ylabel("dB")
ax4b.set_title("Directivity index")
ax4b.grid(True)

fig4.tight_layout()

out_dir = Path.cwd() / "figures"
out_dir.mkdir(exist_ok=True)
for i, fig in enumerate((fig1, fig2, fig3, fig4), start=1):
    fig.savefig(out_dir / f"axi_horn_demo1_fig{i}.png", dpi=150)
print(f"Plots saved to {out_dir.resolve()}/")
