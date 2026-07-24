"""Horn geometry: contour generation and stepped discretization.

MATLAB originals:
  - MMM_1Dhorncoord -> horn_coord_1d
  - MMM_makesteps -> make_steps
"""

import warnings

import numpy as np


def make_steps(coords: np.ndarray) -> np.ndarray:
    """Convert smooth horn profile to stepped duct approximation.

    Corresponds to MMM_makesteps.

    Parameters
    ----------
    coords : (N, D) array
        Smooth horn coordinates. Column 0 is z, columns 1+ are radii.

    Returns
    -------
    newcoords : (M, D) array
        Stepped coordinates.
    """
    N, D = coords.shape

    if D == 3:
        M = N * 4
    else:
        M = N * 2

    result = np.zeros((M, D))
    result[0, :] = coords[0, :]

    for ih in range(N - 1):
        z_mid = coords[ih, 0] + (coords[ih + 1, 0] - coords[ih, 0]) / 2.0
        idx = (ih + 1) * 2  # 0-based: ih*2 → (ih+1)*2

        result[idx - 1, 0] = z_mid
        result[idx - 1, 1:] = coords[ih, 1:]
        result[idx, 0] = z_mid
        result[idx, 1:] = coords[ih + 1, 1:]

    result[-1, :] = coords[-1, :]

    if D == 3:
        result = result[: 2 * N, :]

    return result


def horn_coord_1d(
    htype: str,
    yth: float,
    ym: float,
    length: float,
    tn: float = 1.0,
    dz: float = 0.001,
    add_radius: bool = False,
    th_fta: float = 0.0,
    rad_r: float = 0.1,
    rad_fta: float = 80.0,
) -> np.ndarray:
    """Generate a 1D horn contour profile.

    Corresponds to MMM_1Dhorncoord.

    Returns (N, 2) array: columns are (z, radius).
    """
    L = length

    if L > 0:
        N = int(np.ceil(L / dz))
        dz = L / N
        horncoord = np.zeros((N, 2))
    else:
        N = 0
        horncoord = np.zeros((0, 2))

    htype_lower = htype.lower()

    if yth == abs(ym):
        horncoord[:, 0] = np.linspace(0, L, N)
        horncoord[:, 1] = yth
    else:
        if htype_lower == "conical":
            x1 = L / (ym / yth - 1)
            horncoord[:, 0] = np.linspace(0, L, N)
            horncoord[:, 1] = yth * (horncoord[:, 0] + x1) / x1

        elif htype_lower in ("exponential", "expo"):
            m = np.log(ym / yth) / L
            horncoord[:, 0] = np.linspace(0, L, N)
            horncoord[:, 1] = yth * np.exp(horncoord[:, 0] * m)

        elif htype_lower == "hypex":
            Sth = yth**2 * np.pi
            Sm = ym**2 * np.pi
            tt = (
                -2 * Sth
                + 2 * Sth * tn**2
                + 4 * Sm
                + 4 * np.sqrt(-Sm * Sth + Sm * Sth * tn**2 + Sm**2)
            )
            tn_val = 2 * Sth * (tn * (tn + 2) + 1)
            m = 1.0 / (2 * L) * np.log(tt / tn_val)
            horncoord[:, 0] = np.linspace(0, L, N)
            horncoord[:, 1] = yth * (
                np.cosh(horncoord[:, 0] * m) + tn * np.sinh(horncoord[:, 0] * m)
            )

        elif htype_lower == "oswg":
            rt2 = yth**2
            rm2 = ym**2
            tanang2 = (rm2 - rt2) / L**2
            horncoord[:, 0] = np.linspace(0, L, N)
            horncoord[:, 1] = np.sqrt(rt2 + tanang2 * horncoord[:, 0] ** 2)

        elif htype_lower == "bessel":
            Sth = yth**2 * np.pi
            Sm = ym**2 * np.pi
            x1 = L / ((Sm / Sth) ** (1.0 / tn) - 1)
            horncoord[:, 0] = np.linspace(0, L, N)
            horncoord[:, 1] = yth * ((horncoord[:, 0] + x1) / x1) ** (tn / 2.0)

        elif htype_lower == "spherical":
            Sm = ym**2 * np.pi
            if Sm < 0:
                kc = -Sm
                _, Xhmax = _calc_spherical_max_l(yth, kc)
                if L > Xhmax:
                    L = Xhmax
                    print(
                        f"horn_coord_1d: L too large! Value changed to {L} m"
                    )
                N = int(np.ceil(L / dz))
                dz = L / N
                horncoord = np.zeros((N, 2))
                horncoord[:, 0] = np.linspace(0, L, N)
            else:
                R1 = yth
                R2 = ym
                kcmax = np.log(ym / yth) / L
                kcmin = 0.0
                diff = 1.0
                while abs(diff) > 1e-6:
                    kc = (kcmin + kcmax) / 2.0
                    m_val = 2.0 * kc
                    R0 = 4.0 / m_val
                    H0 = R0 - np.sqrt(R0**2 - R1**2)
                    H = R0 - np.sqrt(R0**2 - R2**2)
                    VX = np.log(H / H0) / m_val
                    if (H0 + VX - H) > L:
                        kcmin = kc
                    else:
                        kcmax = kc
                    diff = kcmax - kcmin

                print(
                    f"Spherical horn cutoff kc = {kc} "
                    f"( = {kc * 344 / 2 / np.pi} Hz)"
                )
                if abs(np.imag(VX)) > 0:
                    print("Invalid horn data - change length!")
                    L = float(np.real(H0 + VX - H))

                N = int(np.ceil(L / dz))
                dz = L / N
                horncoord = np.zeros((N, 2))
                horncoord[:, 0] = np.linspace(0, L, N)

            for ii in range(N):
                horncoord[ii, 1] = _get_spherical_yx(yth, kc, horncoord[ii, 0])

        elif htype_lower == "tractrix":
            if yth > ym:
                warnings.warn(
                    "Throat area must be smaller than mouth area for tractrix horn!"
                )
            else:
                if L < 0:
                    kc = -L
                else:
                    Yx = ym
                    kcmax = 1.0 / Yx
                    kcmin = 0.0
                    Xmin = _tractrix(yth, Yx)
                    if L < Xmin:
                        L = Xmin
                        print(
                            f"horn_coord_1d: L too small! Value changed to {L} m"
                        )
                        kc = kcmax
                    else:
                        diff = 1.0
                        while abs(diff) > 1e-6:
                            kc = (kcmax + kcmin) / 2.0
                            Xmax = _tractrix(yth, 1.0 / kc)
                            XfromM = _tractrix(Yx, 1.0 / kc)
                            tx = Xmax - XfromM
                            diff = tx - L
                            if diff > 0:
                                kcmin = kc
                            else:
                                kcmax = kc

                print(
                    f"Tractrix cutoff kc = {kc} "
                    f"( = {kc * 344 / 2 / np.pi} Hz)"
                )
                ym = 1.0 / kc
                Xmax = _tractrix(yth, ym)
                XfromM = _tractrix(Yx, ym)
                L = float(np.real(Xmax - XfromM))
                N = int(np.ceil(L / dz))
                dz = L / N
                horncoord = np.zeros((N, 2))
                horncoord[:, 0] = np.linspace(0, L, N)
                horncoord[0, 1] = yth
                for ii in range(1, N):
                    horncoord[ii, 1] = _calc_tractrix_at_x(
                        yth, kc, horncoord[ii, 0]
                    )

        elif htype_lower == "radius":
            rad_fta = min(rad_fta, 90.0)
            r0 = yth - rad_r * (1.0 - np.cos(th_fta * np.pi / 180.0))
            z0 = -rad_r * np.sin(th_fta * np.pi / 180.0)
            zL = rad_r * np.sin(rad_fta * np.pi / 180.0)
            L = z0 + zL
            N = int(np.ceil(L / dz))
            dz = L / N  # noqa: F841
            horncoord = np.zeros((N, 2))
            horncoord[:, 0] = np.linspace(0, L, N)
            theta = np.real(np.arcsin((horncoord[:, 0] - z0) / rad_r))
            horncoord[:, 1] = r0 + rad_r - rad_r * np.cos(theta)

        elif htype_lower == "flared conical":
            a = yth
            b = (ym / 1.5 - yth) / L
            c = (ym - b * L - a) / L**tn
            x = np.linspace(0, L, N)
            horncoord[:, 0] = x
            horncoord[:, 1] = a + b * x + c * x**tn

        else:
            horncoord[:, 0] = np.linspace(0, L, N)
            horncoord[:, 1] = yth
            warnings.warn(f"Horn type '{htype}' is not supported.")

    if add_radius:
        maxs = horncoord[-1, 1] ** 2 * np.pi
        fta1 = (
            180.0
            / np.pi
            * np.arctan(
                (horncoord[-1, 1] - horncoord[-2, 1])
                / (horncoord[-1, 0] - horncoord[-2, 0])
            )
        )
        horncoord2 = horn_coord_1d(
            "radius",
            maxs,
            0.0,
            L,
            tn,
            dz,
            add_radius=False,
            th_fta=fta1,
            rad_r=rad_r,
            rad_fta=rad_fta,
        )
        horncoord2[:, 0] = horncoord2[:, 0] + np.max(horncoord[:, 0])
        horncoord = np.vstack([horncoord, horncoord2])

    return horncoord


# ---------------------------------------------------------------------------
# Internal helpers for tractrix and spherical horn contours
# ---------------------------------------------------------------------------


def _calc_tractrix_at_x(Yt: float, kc: float, x: float) -> float:
    Ym = 1.0 / kc
    Xmax = _tractrix(Yt, Ym)
    xfromM = Xmax - x
    Ymax = Ym
    Ymin = Yt
    diff = 1.0
    y = 0.0
    while abs(diff) > 1e-8:
        y = (Ymin + Ymax) / 2.0
        tx = _tractrix(y, Ym)
        diff = tx - xfromM
        if diff > 0:
            Ymin = y
        else:
            Ymax = y
    return y


def _tractrix(Y1: float, Y2: float) -> float:
    return Y2 * np.log((Y2 + np.sqrt(Y2**2 - Y1**2)) / Y1) - np.sqrt(
        Y2**2 - Y1**2
    )


def _calc_spherical_max_l(y0: float, kc: float):
    m = 2.0 * kc
    r0 = 2.0 / kc
    h0 = r0 - np.sqrt(r0**2 - y0**2)
    Xmax = 1.0 / m * np.log(1.0 / (m * h0))
    Xhmax = Xmax + h0 * (1.0 - np.exp(m * Xmax))
    return Xmax, Xhmax


def _get_spherical_vir_length(y0: float, kc: float, Lphys: float) -> float:
    m = 2.0 * kc
    r0 = 2.0 / kc
    h0 = r0 - np.sqrt(r0**2 - y0**2)
    Xvir_min = Lphys
    Xvir_max = 1.0 / m * np.log(1.0 / (m * h0))
    Xhmax = Xvir_max + h0 * (1.0 - np.exp(m * Xvir_max))
    if Lphys > Xhmax:
        return Xvir_max

    diff = 1.0
    Xvir = 0.0
    while abs(diff) > 1e-8:
        Xvir = (Xvir_min + Xvir_max) / 2.0
        diff = Xvir + h0 * (1.0 - np.exp(m * Xvir)) - Lphys
        if diff < 0:
            Xvir_min = Xvir
        else:
            Xvir_max = Xvir
    return Xvir


def _get_spherical_yx(y0: float, kc: float, x: float) -> float:
    Xv = _get_spherical_vir_length(y0, kc, x)
    return _get_spherical_yx_vir(y0, kc, Xv)


def _get_spherical_yx_vir(y0: float, kc: float, xvir: float) -> float:
    m = 2.0 * kc
    r0 = 2.0 / kc
    h0 = r0 - np.sqrt(r0**2 - y0**2)
    h = h0 * np.exp(m * xvir)
    Ak = 2.0 * np.pi * r0 * h
    return np.sqrt(Ak / np.pi - h**2)
