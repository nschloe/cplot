from __future__ import annotations

from typing import Callable

import colorio
import numpy as np
from colorio.cs import ColorCoordinates
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
    colorspace: str = "cam16",
    saturation_adjustment: float = 1.28,
) -> np.ndarray:
    z = np.asarray(z)

    angle = np.arctan2(z.imag, z.real)
    absval_scaled = abs_scaling(np.abs(z))

    # We may have NaNs, so don't be too strict here.
    # assert np.all(absval_scaled >= 0)
    # assert np.all(absval_scaled <= 1)

    if colorspace.upper() == "CAM16":
        # Choose the viewing conditions as "viewing self-luminous display under office
        # illumination".
        cam = colorio.cs.CAM16UCS(c=0.69, Y_b=20, L_A=15)

        # from .create import find_max_srgb_radius
        # r0 = find_max_srgb_radius(cam, L=50)
        # print(r0)
        r0 = 23.545314371585846
        r0 *= saturation_adjustment

        # Rotate the angles such that a "green" color represents positive real values.
        # The rotation offset is chosen such that the ratio g/(r+b) (in rgb) is the
        # largest for the point 1.0.
        #
        # Out of green/red, green is rather associated with "positive value":
        # <https://twitter.com/nschloesoft/status/1452222867152715776>
        # Same for blue in blue vs. orange:
        # <https://twitter.com/nschloesoft/status/1452222679113781249>
        offset = 0.916708 * np.pi
        # Map (r, angle) to a point in the color space; bicone mapping similar to what
        # HSL looks like <https://en.wikipedia.org/wiki/HSL_and_HSV>.
        rd = r0 - r0 * 2 * abs(absval_scaled - 0.5)
        coords = ColorCoordinates(
            [
                100 * absval_scaled,
                rd * np.cos(angle + offset),
                rd * np.sin(angle + offset),
            ],
            cam,
        )
        srgb_vals = coords.get_rgb1("clip")

    elif colorspace.upper() == "CIELAB":
        cielab = colorio.cs.CIELAB()

        # from .create import find_max_srgb_radius
        # r0 = find_max_srgb_radius(cielab, L=50)
        # print(r0)
        # exit(1)
        r0 = 29.488203674554825
        r0 *= saturation_adjustment

        # Rotate the angles such that a "green" color represents positive real values.
        # The rotation is chosen such that the ratio g/(r+b) (in rgb) is the largest for
        # the point 1.0.
        offset = 0.8936868 * np.pi
        # Map (r, angle) to a point in the color space; bicone mapping similar to what
        # HSL looks like <https://en.wikipedia.org/wiki/HSL_and_HSV>.
        rd = r0 - r0 * 2 * abs(absval_scaled - 0.5)
        coords = ColorCoordinates(
            [
                100 * absval_scaled,
                rd * np.cos(angle + offset),
                rd * np.sin(angle + offset),
            ],
            cielab,
        )
        srgb_vals = coords.get_rgb1("clip")

    elif colorspace.upper() == "OKLAB":
        oklab = colorio.cs.OKLAB()
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
        coords = ColorCoordinates(
            [
                # OKLAB is designed such that the D65 whitepoint, scaled to Y=100, has
                # lightness 1.
                absval_scaled,
                rd * np.cos(angle + offset),
                rd * np.sin(angle + offset),
            ],
            oklab,
        )
        srgb_vals = coords.get_rgb1("clip")

    else:
        assert (
            colorspace.upper() == "HSL"
        ), f"Illegal colorspace {colorspace}. Pick one of CAM16, CIELAB, HSL."
        hsl = colorio.cs.HSL()
        # rotate by 120 degrees to have green (0, 1, 0) for real positive numbers
        offset = 120
        coords = np.array(
            [
                np.mod(angle / (2 * np.pi) * 360 + offset, 360),
                np.ones(angle.shape),
                absval_scaled,
            ]
        )
        srgb_vals = hsl.to_rgb1(coords)

    return np.moveaxis(srgb_vals, 0, -1)
