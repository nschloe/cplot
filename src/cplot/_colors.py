from __future__ import annotations

from typing import Callable

import npx
import numpy as np
from numpy.typing import ArrayLike

# Lightness: |f| -> s in [0, 1]
# -----------------------------
# The scaling is fixed (no parameters) so that plots of different functions are
# directly comparable: the same lightness always means the same |f|. It is
#
#   s = σ(asinh(ln(r) / 2)),   σ(t) = 1 / (1 + exp(-t)),
#
# or, equivalently, the classical r / (r + 1) applied to a log-compressed r,
#
#   s = w / (w + 1),   w = y + sqrt(1 + y²),   y = ln(r) / 2.
#
# Properties:
#
#  * s(1/r) = 1 - s(r), so the inverse of a function is exactly as light as the
#    original function is dark. s(1) = 1/2.
#
#  * Near r = 1, w ≈ r^(1/2), so the scaling behaves like r^a / (r^a + 1) with
#    a = 1/2.
#
#  * Far from r = 1, w only grows like ln(r), so the tails are fat:
#    1 - s ~ 1 / |ln r| instead of 1 - s ~ r^(-a). Lightness keeps changing over many
#    orders of magnitude, and pure black / pure white (in 8-bit sRGB) are effectively
#    reserved for exactly 0 / infinity; the first pure-black pixel appears at
#    |f| ≈ 1e-36.
#
#  * There is no sharp knee in the curve; steeper slopes at r = 1 combined with fat
#    tails lead to visible edges around |f| = 1.
#
# s is used as the toe-corrected OKLab lightness L_r (Ottosson,
# <https://bottosson.github.io/posts/colorpicker/#intermission---a-new-lightness-estimate-for-oklab>),
# which is closer to perceived lightness than OKLab's L, in particular for dark
# colors.
#
# Hue: arg -> hue
# ---------------
# The hue is the argument rotated such that positive real values are green, plus a
# small periodic correction that makes neighboring arguments equally far apart (in
# CAM16-UCS) on the circle |f| = 1. The correction is tabulated in _HUE_CORRECTION
# at 64 equidistant arguments in [-pi, pi). arg 0 is not moved.


def scale_abs(r: ArrayLike) -> np.ndarray:
    """Map |f| in [0, infinity] to lightness s in [0, 1], s(1/r) = 1 - s(r)."""
    r = np.asarray(r, dtype=float)
    with np.errstate(divide="ignore"):
        x = np.log(r)
    # σ(t) = (1 + tanh(t / 2)) / 2 is numerically stable for all t
    return 0.5 * (1.0 + np.tanh(0.5 * np.arcsinh(0.5 * x)))


def scale_abs_inv(s: ArrayLike) -> np.ndarray:
    """Inverse of scale_abs."""
    s = np.asarray(s, dtype=float)
    with np.errstate(divide="ignore", over="ignore"):
        return np.exp(2.0 * np.sinh(np.log(s / (1.0 - s))))


def _toe_inv(lr: np.ndarray) -> np.ndarray:
    # Ottosson's toe, inverse: toe-corrected lightness L_r -> OKLab L
    k1 = 0.206
    k2 = 0.03
    k3 = (1.0 + k1) / (1.0 + k2)
    return (lr * lr + k1 * lr) / (k3 * (lr + k2))


# Rotate the angles such a "green" color represents positive real values. The rotation
# is chosen such that the ratio g/(r+b) (in rgb) is the largest for the point 1.0.
_HUE_OFFSET = 0.8936868 * np.pi

_HUE_T = np.linspace(-np.pi, np.pi, 64, endpoint=False)
# fmt: off
_HUE_CORRECTION = np.array([
    -0.05253, -0.04273, -0.03096, -0.01760, -0.00327, +0.01116, +0.02472, +0.03652,
    +0.04588, +0.05245, +0.05621, +0.05739, +0.05643, +0.05383, +0.05018, +0.04605,
    +0.04199, +0.03844, +0.03568, +0.03374, +0.03240, +0.03127, +0.02992, +0.02807,
    +0.02557, +0.02249, +0.01897, +0.01524, +0.01154, +0.00806, +0.00496, +0.00228,
    +0.00000, -0.00198, -0.00384, -0.00081, -0.00009, -0.00311, -0.00862, -0.01534,
    -0.02191, -0.02690, -0.02867, -0.02533, -0.01918, -0.02511, -0.03090, -0.03644,
    -0.04163, -0.04644, -0.05084, -0.05484, -0.05846, -0.06175, -0.06470, -0.06732,
    -0.06956, -0.07131, -0.07242, -0.07269, -0.07188, -0.06971, -0.06591, -0.06024,
])
# fmt: on


def get_srgb1(
    z: ArrayLike,
    abs_scaling: Callable[[np.ndarray], np.ndarray] | None = None,
    saturation_adjustment: float = 1.28,
) -> np.ndarray:
    z = np.asarray(z)

    angle = np.arctan2(z.imag, z.real)
    absval = np.abs(z)
    absval_scaled = scale_abs(absval) if abs_scaling is None else abs_scaling(absval)

    # We may have NaNs, so don't be too strict here.
    # assert np.all(absval_scaled >= 0)
    # assert np.all(absval_scaled <= 1)

    # from .create import find_max_srgb_radius
    # r0 = find_max_srgb_radius(oklab, L=0.5)
    r0 = 0.08499547839164734
    r0 *= saturation_adjustment

    hue = (
        angle
        + np.interp(angle, _HUE_T, _HUE_CORRECTION, period=2 * np.pi)
        + _HUE_OFFSET
    )
    # Map (r, angle) to a point in the color space; bicone mapping similar to what
    # HSL looks like <https://en.wikipedia.org/wiki/HSL_and_HSV>.
    rd = r0 - r0 * 2 * abs(absval_scaled - 0.5)
    ok_coords = np.array(
        [
            _toe_inv(absval_scaled),
            rd * np.cos(hue),
            rd * np.sin(hue),
        ]
    )
    xyz100 = oklab_to_xyz100(ok_coords)
    srgb1 = xyz100_to_srgb1(xyz100)

    return np.moveaxis(srgb1, 0, -1)


def oklab_to_xyz100(lab: np.ndarray) -> np.ndarray:
    M1 = np.array(
        [
            [0.8189330101, 0.3618667424, -0.1288597137],
            [0.0329845436, 0.9293118715, 0.0361456387],
            [0.0482003018, 0.2643662691, 0.6338517070],
        ]
    )
    M1inv = np.linalg.inv(M1)
    M2 = np.array(
        [
            [0.2104542553, +0.7936177850, -0.0040720468],
            [+1.9779984951, -2.4285922050, +0.4505937099],
            [+0.0259040371, +0.7827717662, -0.8086757660],
        ]
    )
    M2inv = np.linalg.inv(M2)
    return npx.dot(M1inv, npx.dot(M2inv, lab) ** 3) * 100


def _xyy_to_xyz100(xyy: np.ndarray) -> np.ndarray:
    x, y, Y = xyy
    return np.array([Y / y * x, Y, Y / y * (1 - x - y)]) * 100


def xyz100_to_srgb_linear(xyz: np.ndarray) -> np.ndarray:
    primaries_xyy = np.array(
        [
            [0.64, 0.33, 0.2126],
            [0.30, 0.60, 0.7152],
            [0.15, 0.06, 0.0722],
        ]
    )
    invM = _xyy_to_xyz100(primaries_xyy.T)
    whitepoint_correction = True
    if whitepoint_correction:
        # The above values are given only approximately, resulting in the fact that
        # SRGB(1.0, 1.0, 1.0) is only approximately mapped into the reference
        # whitepoint D65. Add a correction here.
        whitepoints_cie1931_d65 = np.array([95.047, 100, 108.883])
        correction = whitepoints_cie1931_d65 / np.sum(invM, axis=1)
        invM = (invM.T * correction).T
    invM /= 100

    # https://en.wikipedia.org/wiki/SRGB#The_forward_transformation_(CIE_XYZ_to_sRGB)
    # https://www.color.org/srgb.pdf
    out = npx.solve(invM, xyz) / 100
    out = out.clip(0.0, 1.0)
    return out


def xyz100_to_srgb1(xyz: np.ndarray) -> np.ndarray:
    srgb = xyz100_to_srgb_linear(xyz)
    # gamma correction:
    a = 0.055
    is_smaller = srgb <= 0.0031308
    srgb[is_smaller] *= 12.92
    srgb[~is_smaller] = (1 + a) * srgb[~is_smaller] ** (1 / 2.4) - a
    return srgb
