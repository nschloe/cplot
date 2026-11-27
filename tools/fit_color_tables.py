"""Fit cplot's two hue correction tables (_HUE_CORRECTION, _LIGHTNESS_CORRECTION).

On the circle |f| = 1, cplot's colors have OKLab lightness toe_inv(0.6) and constant
chroma. This script finds, at 64 equidistant arguments,

  * a hue offset such that neighboring arguments are equally far apart in CAM16-UCS
    (even "speed" of color change around the circle); arg 0 stays in place;
  * an OKLab lightness offset such that all hues have the same perceived lightness
    (Hellwig 2022 J_HK, which includes the Helmholtz-Kohlrausch effect).

Both depend on each other, so they are fitted alternately. Rerun this when the hue
offset, the lightness at |f| = 1, or the chroma in cplot/_colors.py changes, and paste
the printed tables into _colors.py.

    uv run --with colour-science tools/fit_color_tables.py
"""

import colour
import numpy as np

from cplot._colors import (
    _HUE_OFFSET,
    _LIGHTNESS_EXPONENT,
    _toe_inv,
    oklab_to_xyz100,
    xyz100_to_srgb1,
)

CHROMA = 0.08499547839164734 * 1.28  # r0 * default saturation_adjustment
L0 = _toe_inv(0.5**_LIGHTNESS_EXPONENT)  # OKLab L at |f| = 1
M = 64
T = np.linspace(-np.pi, np.pi, M, endpoint=False)

XYZ_W = (
    colour.xy_to_XYZ(
        colour.CCS_ILLUMINANTS["CIE 1931 2 Degree Standard Observer"]["D65"]
    )
    * 100
)
# typical office viewing: 64 lux adapting field, 20% grey background
VC = dict(XYZ_w=XYZ_W, L_A=64 / np.pi * 0.2, Y_b=20)


def appearance(rgb):
    """sRGB1 -> (Hellwig 2022 J_HK, CAM16-UCS J'a'b')."""
    with colour.utilities.suppress_warnings(colour_usage_warnings=True):
        XYZ = colour.sRGB_to_XYZ(rgb) * 100
        hw = colour.XYZ_to_Hellwig2022(
            XYZ, surround=colour.VIEWING_CONDITIONS_HELLWIG2022["Average"], **VC
        )
        c16 = colour.XYZ_to_CAM16(
            XYZ, surround=colour.VIEWING_CONDITIONS_CAM16["Average"], **VC
        )
        ucs = colour.JMh_CAM16_to_CAM16UCS(np.stack([c16.J, c16.M, c16.h], axis=-1))
    return hw.J_HK, ucs


def colors(arg, dh, dl):
    h = arg + np.interp(arg, T, dh, period=2 * np.pi) + _HUE_OFFSET
    L = L0 + np.interp(arg, T, dl, period=2 * np.pi)
    lab = np.array([L, CHROMA * np.cos(h), CHROMA * np.sin(h)])
    return np.clip(np.moveaxis(xyz100_to_srgb1(oklab_to_xyz100(lab)), 0, -1), 0, 1)


def periodic(x, y):
    return (
        np.concatenate([x - 2 * np.pi, x, x + 2 * np.pi]),
        np.concatenate([y - 2 * np.pi, y, y + 2 * np.pi]),
    )


def fit(iterations=4):
    a = np.linspace(-np.pi, np.pi, 4097)
    dh = np.zeros(M)
    dl = np.zeros(M)
    for _ in range(iterations):
        # even speed: reparametrize the circle by CAM16-UCS arc length
        _, ucs = appearance(colors(a, dh, dl))
        arc = np.concatenate(
            [[0], np.cumsum(np.linalg.norm(np.diff(ucs, axis=0), axis=1))]
        )
        target = -np.pi + arc / arc[-1] * 2 * np.pi
        target -= np.interp(0.0, a, target)  # keep arg 0 fixed
        hue = a + np.interp(a, T, dh, period=2 * np.pi)
        a_of_t = np.interp(T, *periodic(target, a))
        dh = np.interp(T, *periodic(target, hue)) - T
        dl = np.interp(a_of_t, T, dl, period=2 * np.pi)  # carry lightness along
        # equal perceived lightness: fixed-point iteration on J_HK
        J_target = appearance(colors(T, dh, np.zeros(M)))[0].mean()
        for _ in range(40):
            J = appearance(colors(T, dh, dl))[0]
            dl += (J_target - J) * 0.008
    return dh, dl


def report(dh, dl):
    x = np.linspace(-np.pi, np.pi, 1440, endpoint=False)
    J, ucs = appearance(colors(x, dh, dl))
    step = np.linalg.norm(np.roll(ucs, -1, axis=0) - ucs, axis=1)
    step /= step.mean()
    w = 20  # 5° windows
    smooth = np.convolve(np.concatenate([step[-w:], step, step[:w]]), np.ones(w) / w)
    smooth = smooth[2 * w - 1 : -w]
    print(f"speed (5° windows): {smooth.min():.3f}-{smooth.max():.3f}")
    print(f"J_HK: {J.min():.2f}-{J.max():.2f}")


def fmt(name, v):
    rows = [", ".join(f"{q:+.5f}" for q in v[i : i + 8]) for i in range(0, len(v), 8)]
    return f"{name} = np.array([\n" + "".join(f"    {r},\n" for r in rows) + "])"


if __name__ == "__main__":
    dh, dl = fit()
    report(dh, dl)
    print(fmt("_HUE_CORRECTION", dh))
    print(fmt("_LIGHTNESS_CORRECTION", dl))
