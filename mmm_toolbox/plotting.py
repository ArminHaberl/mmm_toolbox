"""Visualization and post-processing: directivity index, polar maps.

MATLAB originals:
  - MMM_ASgetDI -> get_di_axi
"""

import warnings

import numpy as np


def get_di_axi(data: dict, angles: np.ndarray) -> dict:
    """Compute directivity index from radiated pressure field.

    Corresponds to MMM_ASgetDI.

    Modifies data dict in-place: sets data['DI'].
    Returns data for convenience.
    """
    p_mag = np.abs(data["pRad"])
    n_angles = len(angles)

    if n_angles > 100:
        dtheta = np.pi / 180.0 * angles[1]
        sia = dtheta * np.sin(np.pi / 180.0 * angles)
        sia = sia[:, np.newaxis]  # (n_angles, 1)
        Wrad = np.sum(p_mag**2 * sia, axis=0)
        Q = 2.0 * (p_mag[0, :] ** 2) / Wrad
    else:
        dang = angles[1]
        n = round(180.0 / dang)
        m = n // 2 + 1

        wt = np.zeros(m)
        for r in range(0, n + 1, 2):
            k1 = 0.5 if (r == 0 or r == n) else 1.0
            wt[0] = wt[0] + k1 * (-1.0 / (r**2 - 1.0))
        wt[0] = wt[0] / n

        for i in range(1, m):
            for r in range(0, n + 1, 2):
                k1 = 0.5 if (r == 0 or r == n) else 1.0
                wt[i] = wt[i] + k1 * (-1.0 / (r**2 - 1.0)) * np.cos(
                    np.pi * r * i / n
                )
            wt[i] = 2.0 * wt[i] / n

        wt = wt[:, np.newaxis]  # (m, 1)
        Q = np.sum(wt * p_mag[:m, :] ** 2, axis=0)
        Q = p_mag[0, :] ** 2 / Q

        maxq = (0.5 * n + 1) ** 2 * np.sqrt(2.0)
        if np.max(Q) > maxq:
            warnings.warn(
                f"Directivity index larger than "
                f"{10 * np.log10(maxq):.0f} dB is unreliable. "
                f"Please use more field points."
            )

    data["DI"] = 10.0 * np.log10(np.abs(Q))
    return data
