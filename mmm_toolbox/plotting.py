"""Visualization and post-processing: directivity index, polar maps,
horn profile, throat impedance, SPL, and sound-field plots.

MATLAB originals:
  - MMM_ASgetDI -> get_di_axi
  - MMM_ASplotHorn -> plot_horn_profile
  - MMM_PlotZth -> plot_throat_impedance
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

        r = np.arange(0, n + 1, 2)
        k1 = np.where((r == 0) | (r == n), 0.5, 1.0)
        term = k1 * (-1.0 / (r**2 - 1.0))

        wt = np.zeros(m)
        wt[0] = np.sum(term) / n

        if m > 1:
            i_vals = np.arange(1, m)
            cos_mat = np.cos(np.pi * np.outer(r, i_vals) / n)
            wt[1:] = 2.0 / n * (term @ cos_mat)

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


# ---------------------------------------------------------------------------
# Plotting functions (require matplotlib)
# ---------------------------------------------------------------------------


def _ensure_ax(ax, figsize=None):
    """If *ax* is None, create a new figure + axes; otherwise return existing."""
    import matplotlib.pyplot as plt

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.get_figure()
    return fig, ax


def plot_horn_profile(data: dict, ax=None):
    """Plot the raw and stepped horn contour (axisymmetric).

    Parameters
    ----------
    data : dict
        Initialized horn data dict (must contain ``raw_coords`` and
        ``stepped_coords``).
    ax : matplotlib.axes.Axes, optional
        Axes to draw on.  Created if not given.

    Returns
    -------
    fig : matplotlib.figure.Figure
    ax : matplotlib.axes.Axes
    """
    fig, ax = _ensure_ax(ax, figsize=(8, 5))

    rc = data["raw_coords"]
    sc = data["stepped_coords"]
    ym = np.max(rc[:, 1])

    ax.plot(sc[:, 0], sc[:, 1], "b",
            sc[:, 0], -sc[:, 1], "b",
            rc[:, 0], rc[:, 1], "k",
            rc[:, 0], -rc[:, 1], "k")
    ax.set_ylim(-ym * 1.1, ym * 1.1)
    ax.set_aspect("equal")
    ax.set_xlabel("z axis [m]")
    ax.set_ylabel("Radius [m]")
    ax.set_title("Horn profile")
    fig.tight_layout()
    return fig, ax


def plot_throat_impedance(data: dict, ax=None):
    """Plot normalised throat impedance vs frequency.

    Parameters
    ----------
    data : dict
        Must contain ``fvec``, ``Z00``, ``St``, ``rho``, ``c``.
    ax : matplotlib.axes.Axes, optional

    Returns
    -------
    fig : matplotlib.figure.Figure
    ax : matplotlib.axes.Axes
    """
    fig, ax = _ensure_ax(ax, figsize=(8, 5))

    Z00n = data["St"] / (data["rho"] * data["c"]) * data["Z00"]

    ax.semilogx(data["fvec"], np.real(Z00n), "k", label="Real")
    ax.semilogx(data["fvec"], np.imag(Z00n), "r", label="Imag")
    ax.set_xlim(data["fvec"][0], data["fvec"][-1])
    ax.set_xlabel("Hz")
    ax.set_ylabel("Normalized acoustic Z")
    ax.set_title("Horn throat impedance")
    ax.legend()
    ax.grid(True)
    fig.tight_layout()
    return fig, ax


def plot_spl_vs_frequency(
    data: dict, angles: np.ndarray, step: float = 10.0, ax=None,
):
    """Plot SPL vs frequency for field points at multiples of *step* degrees.

    Parameters
    ----------
    data : dict
        Must contain ``fvec`` and ``pRad``.
    angles : (n_angles,) array
        Field-point angles in degrees (must match ``pRad`` rows).
    step : float
        Angle spacing between plotted curves (default 10°).
    ax : matplotlib.axes.Axes, optional

    Returns
    -------
    fig : matplotlib.figure.Figure
    ax : matplotlib.axes.Axes
    """
    fig, ax = _ensure_ax(ax, figsize=(9, 6))

    ia = np.where(np.mod(angles, step) == 0)[0]
    SPL = 94.0 + 20.0 * np.log10(np.abs(data["pRad"]))

    for idx in ia:
        ax.semilogx(data["fvec"], SPL[idx, :], label=f"{angles[idx]:.0f}°")

    ax.set_xlim(data["fvec"][0], data["fvec"][-1])
    ax.set_xlabel("Hz")
    ax.set_ylabel("dB SPL")
    ax.set_title("Field point pressures")
    ax.legend(title="Angle", ncol=2, fontsize="small")
    ax.grid(True)
    fig.tight_layout()
    return fig, ax


def plot_polar_map(data: dict, angles: np.ndarray, ax=None):
    """Plot a normalised polar SPL map (contour over frequency vs angle).

    Parameters
    ----------
    data : dict
        Must contain ``fvec`` and ``pRad``.
    angles : (n_angles,) array
        Field-point angles in degrees.
    ax : matplotlib.axes.Axes, optional

    Returns
    -------
    fig : matplotlib.figure.Figure
    ax : matplotlib.axes.Axes
    """
    fig, ax = _ensure_ax(ax, figsize=(9, 4.5))

    Lp = 20.0 * np.log10(np.abs(data["pRad"]))
    Lp_norm = Lp - Lp[0, :]

    cf = ax.contourf(data["fvec"], angles, Lp_norm, 15, cmap="viridis")
    ax.set_xscale("log")
    ax.set_xlim(data["fvec"][0], data["fvec"][-1])
    ax.set_xlabel("Hz")
    ax.set_ylabel("Degrees")
    ax.set_title("Polar map (normalized)")
    fig.colorbar(cf, ax=ax, label="dB")
    fig.tight_layout()
    return fig, ax


def plot_directivity_index(data: dict, ax=None):
    """Plot directivity index vs frequency.

    Parameters
    ----------
    data : dict
        Must contain ``fvec`` and ``DI`` (call :func:`get_di_axi` first).
    ax : matplotlib.axes.Axes, optional

    Returns
    -------
    fig : matplotlib.figure.Figure
    ax : matplotlib.axes.Axes
    """
    fig, ax = _ensure_ax(ax, figsize=(9, 3))

    ax.semilogx(data["fvec"], data["DI"], "k")
    ax.set_xlim(data["fvec"][0], data["fvec"][-1])
    ax.set_xlabel("Hz")
    ax.set_ylabel("dB")
    ax.set_title("Directivity index")
    ax.grid(True)
    fig.tight_layout()
    return fig, ax


def plot_sound_field(
    data: dict,
    freq: float | None = None,
    resolution: int = 30,
    add_nearfield: bool = True,
    ax=None,
):
    """Plot the internal (and optionally near-field) SPL distribution.

    Wraps :func:`mmm_toolbox.radiation.pressure_distribution_axi` and
    displays the result as a filled contour map with horn outline overlay.

    Parameters
    ----------
    data : dict
        Initialized horn data dict.
    freq : float, optional
        Single frequency [Hz].  Defaults to ``data["fvec"][0]``.
    resolution : int
        Number of radial sample points (default 30).
    add_nearfield : bool
        If True, append a near-field region in front of the horn.
    ax : matplotlib.axes.Axes, optional

    Returns
    -------
    fig : matplotlib.figure.Figure
    ax : matplotlib.axes.Axes
    """
    from mmm_toolbox.radiation import pressure_distribution_axi

    if freq is None:
        freq = float(data["fvec"].flat[0])

    plotcoordsz, plotcoordsx, Pmatx = pressure_distribution_axi(
        freq, data, add_nearfield=add_nearfield, resolution=resolution,
    )

    fig, ax = _ensure_ax(ax, figsize=(10, 6))

    SPL = 94.0 + 20.0 * np.log10(np.abs(Pmatx))
    cf = ax.contourf(plotcoordsz.T, plotcoordsx.T, SPL.T, 25, cmap="viridis")
    fig.colorbar(cf, ax=ax, label="dB SPL")
    ax.set_aspect("equal")

    rc = data["raw_coords"]
    coords_double = np.vstack((rc, rc[-1, :]))
    coords_double[-1, 1] = 0.0
    ax.plot(coords_double[:, 0], coords_double[:, 1], "k", linewidth=1.5)
    ax.plot(coords_double[:, 0], -coords_double[:, 1], "k", linewidth=1.5)

    n_modes = data["n_modes"]
    ax.set_title(f"Sound field at {freq:.0f} Hz, using {n_modes} modes")
    ax.set_xlabel("z [m]")
    ax.set_ylabel("Radius [m]")
    fig.tight_layout()
    return fig, ax
