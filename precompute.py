"""Precompute the modal radiation impedance lookup table.

Equivalent to running MMM_ASbaffledradzmatrixPrecompute(32) in MATLAB.
Generates ZradAS32.mat via direct numerical integration.

WARNING: This takes several minutes. For 32 modes, expect ~5-15 minutes
depending on your CPU.
"""

from hornsim.radiation import precompute_rad_zmatrix

if __name__ == "__main__":
    precompute_rad_zmatrix(max_modes=2)
