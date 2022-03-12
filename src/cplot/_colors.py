from __future__ import annotations

from typing import Callable

import npx
import numpy as np
from numpy.typing import ArrayLike


# A number of scalings f that map the magnitude [0, infty] to [0, 1] are possible.  One
# desirable property is
#
# (1)  f(1/r) = 1 - f(r).
#
# This makes sure that the representation of the inverse of a function is exactly as
# light as the original function is dark. The function g_a(r) = 1 - a^r (with some 0 < a
# < 1), as it is sometimes suggested (e.g., on Wikipedia
# <https://en.wikipedia.org/wiki/Domain_coloring>) does _not_ fulfill (1).
#
# A common alternative is
#
#   h(r) = r^a / (r^a + 1)
#
# with a configurable parameter a.
#
#  * For a=1.21268891, this function is very close to the popular alternative 2/pi *
#  arctan(r) (which also fulfills the above property)
#
#  * For a=1.21428616 is is close to g_{1/2} (between 0 and 1).
#
#  * For a=1.49486991 it is close to x/2 (between 0 and 1).
#
#  * For a=2 it is the stereographic projection onto the Riemann sphere.
#    For other a, it's the projection onto something else than a sphere. For a=1, for
#    example, the projection onto the line f(t) = t-1.
#
# Disadvantage of this choice:
#
#  h'(r) = (a * r^{a-1} * (r^a + 1) - r^a * a * r^{a-1}) / (r^a + 1) ** 2
#        = a * r^{a-1} / (r^a + 1) ** 2
#
# so h'(r)=0 at r=0 for all a > 1. This means that h(r) has an inflection point in (0,
# 1) for all a > 1. For 0 < alpha < 1, the derivative at 0 is infty.
#
# Only for a=1, the derivative is 1/2. For arctan, it's 1 / pi.
#
# There are other choices of the form f(x) = s(x) / (s(x) + 1). s has to
# fulfill s(x)*s(1/x) = 1, see <https://math.stackexchange.com/a/221415/36678>
# for a characterization of such functions. This leads to the characterization
#
#  f(x) = phi(x) / (phi(x) + phi(1/x))
#
# for _any_ phi(x). For phi(x)=sqrt(x), one gets f(x)=x/(x+1). Ideas here are
# phi = log1p or log1p(log1p) (and so forth).
#
# Another choice that fulfills (1) is
#
#           / r / 2          for 0 <= x <= 1,
#   f(r) = |
#           \ 1 - 1 / (2r)   for x > 1,
#
# but its second derivative is discontinuous at 1, and one does actually notice this
#
#            / 1/2         for 0 <= x <= 1,
#   f'(r) = |
#            \ 1/2 / r^2   for x > 1,
#
#             / 0          for 0 <= x <= 1,
#   f''(r) = |
#             \ -1 / r^3   for x > 1,
#
# TODO find parametrized function that is free of inflection points for the param=0
# (or infty) is this last f(r).
#
def get_srgb1(
    z: ArrayLike,
    abs_scaling: Callable[[np.ndarray], np.ndarray] = lambda x: x / (x + 1),
    saturation_adjustment: float = 1.28,
) -> np.ndarray:
    z = np.asarray(z)

    angle = np.arctan2(z.imag, z.real)
    absval_scaled = abs_scaling(np.abs(z))

    # We may have NaNs, so don't be too strict here.
    # assert np.all(absval_scaled >= 0)
    # assert np.all(absval_scaled <= 1)

    # from .create import find_max_srgb_radius
    # r0 = find_max_srgb_radius(oklab, L=0.5)
    r0 = 0.08499547839164734
    r0 *= saturation_adjustment

    # Rotate the angles such a "green" color represents positive real values. The
    # rotation is chosen such that the ratio g/(r+b) (in rgb) is the largest for the
    # point 1.0.
    offset = 0.8936868 * np.pi
    # Map (r, angle) to a point in the color space; bicone mapping similar to what
    # HSL looks like <https://en.wikipedia.org/wiki/HSL_and_HSV>.
    rd = r0 - r0 * 2 * abs(absval_scaled - 0.5)
    ok_coords = np.array(
        [
            absval_scaled,
            rd * np.cos(angle + offset),
            rd * np.sin(angle + offset),
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
