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
#    |f| ≈ 1e-175, the first pure-white one at |f| ≈ 1e204.
#
#  * There is no sharp knee in the curve; steeper slopes at r = 1 combined with fat
#    tails lead to visible edges around |f| = 1.
#
# s is mapped to the toe-corrected OKLab lightness L_r (Ottosson,
# <https://bottosson.github.io/posts/colorpicker/#intermission---a-new-lightness-estimate-for-oklab>),
# which is closer to perceived lightness than OKLab's L, in particular for dark
# colors, via L_r = s^p with p = ln(0.6) / ln(0.5). This keeps 0 -> black and
# infinity -> white, but puts |f| = 1 at L_r = 0.6 instead of 0.5: colors are a bit
# lighter, which helps the warm hues (orange, yellow) that look brown/olive when dark.
#
# Hue: arg -> hue
# ---------------
# Positive real values are red (the OKLab hue of sRGB red, 29°); the hue runs in the
# conventional direction red -> yellow -> green -> cyan -> blue -> magenta, so
# +i is yellow-olive, -1 teal, -i violet.
#
# Two small periodic corrections, tabulated at 64 equidistant arguments in [-pi, pi)
# and fitted on the circle |f| = 1 (with colour-science's CAM16 and Hellwig 2022
# models, average surround):
#
#  * _HUE_CORRECTION makes neighboring arguments equally far apart in CAM16-UCS
#    (even "speed" of color change); arg 0 is not moved.
#
#  * _LIGHTNESS_CORRECTION is added to OKLab L such that all hues have the same
#    perceived lightness (Hellwig 2022 J_HK, which includes the Helmholtz-Kohlrausch
#    effect). It is scaled with the chroma, so it vanishes for black, white, and grey.
#
# Both are linearly interpolated (and not a smooth fit) since they have kinks where
# the color circle touches the sRGB gamut.


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


_HUE_OFFSET = np.deg2rad(29.0)
_LIGHTNESS_EXPONENT = np.log(0.6) / np.log(0.5)


def lightness_r(s: ArrayLike) -> np.ndarray:
    """Toe-corrected OKLab lightness L_r that get_srgb1 uses for scaled |f| = s."""
    return np.asarray(s, dtype=float) ** _LIGHTNESS_EXPONENT


def srgb1_grey(lr: ArrayLike) -> np.ndarray:
    """sRGB value (in [0, 1]) of the grey with toe-corrected OKLab lightness lr."""
    lr = np.asarray(lr, dtype=float)
    lab = np.array([_toe_inv(lr), np.zeros_like(lr), np.zeros_like(lr)])
    return xyz100_to_srgb1(oklab_to_xyz100(lab))[1]


_CORRECTION_T = np.linspace(-np.pi, np.pi, 64, endpoint=False)
# fmt: off
_HUE_CORRECTION = np.array([
    -0.02465, -0.02903, -0.03468, -0.04063, -0.04675, -0.05295, -0.05916, -0.06529,
    -0.07127, -0.07703, -0.08252, -0.08773, -0.09264, -0.09729, -0.10168, -0.10576,
    -0.10945, -0.11262, -0.11507, -0.11656, -0.11683, -0.11558, -0.11255, -0.10750,
    -0.10029, -0.09089, -0.07945, -0.06635, -0.05216, -0.03765, -0.02365, -0.01091,
    +0.00000, +0.00881, +0.01552, +0.02035, +0.02366, +0.02587, +0.02742, +0.02865,
    +0.02983, +0.03110, +0.03245, +0.03378, +0.03490, +0.03558, +0.03565, +0.03495,
    +0.03343, +0.03109, +0.02798, +0.02421, +0.01994, +0.01538, +0.01076, +0.00629,
    +0.00215, -0.00153, -0.00476, -0.00766, -0.01044, -0.01334, -0.01720, -0.02150,
])
_LIGHTNESS_CORRECTION = np.array([
    -0.02474, -0.02335, -0.02193, -0.02060, -0.01949, -0.01869, -0.01820, -0.01794,
    -0.01782, -0.01773, -0.01753, -0.01716, -0.01654, -0.01565, -0.01450, -0.01312,
    -0.01154, -0.00981, -0.00799, -0.00613, -0.00428, -0.00249, -0.00079, +0.00078,
    +0.00220, +0.00348, +0.00465, +0.00574, +0.00683, +0.00798, +0.00926, +0.01072,
    +0.01238, +0.01426, +0.01631, +0.01850, +0.02076, +0.02299, +0.02508, +0.02693,
    +0.02843, +0.02950, +0.03008, +0.03015, +0.02970, +0.02872, +0.02721, +0.02518,
    +0.02261, +0.01950, +0.01587, +0.01175, +0.00724, +0.00245, -0.00246, -0.00733,
    -0.01197, -0.01619, -0.01980, -0.02267, -0.02468, -0.02589, -0.02637, -0.02593,
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
        + np.interp(angle, _CORRECTION_T, _HUE_CORRECTION, period=2 * np.pi)
        + _HUE_OFFSET
    )
    # Map (r, angle) to a point in the color space; bicone mapping similar to what
    # HSL looks like <https://en.wikipedia.org/wiki/HSL_and_HSV>.
    chroma_weight = 1.0 - 2.0 * abs(absval_scaled - 0.5)
    rd = r0 * chroma_weight
    lightness = _toe_inv(lightness_r(absval_scaled)) + (
        chroma_weight
        * (saturation_adjustment / 1.28)
        * np.interp(angle, _CORRECTION_T, _LIGHTNESS_CORRECTION, period=2 * np.pi)
    )
    ok_coords = np.array(
        [
            lightness,
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
