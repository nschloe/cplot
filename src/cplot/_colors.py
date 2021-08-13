import colorio
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt


def scale01(vals: npt.ArrayLike, scaling_type: str):
    vals = np.asarray(vals)
    assert np.all(vals >= 0)

    if scaling_type == "arctan":
        # Fulfills (1), but not parametrized.
        # def f_inv(y):
        #     return np.tan(np.pi / 2 * y)
        return 2 / np.pi * np.arctan(vals)

    # Fulfills (1) for any alpha >= 0
    # def f_inv(y):
    #     return (y / (1 - y)) ** (1 / alpha)
    assert scaling_type.startswith("h-")
    alpha = float(scaling_type[2:])
    assert alpha >= 0
    return vals ** alpha / (vals ** alpha + 1)


def get_srgb1(z: npt.ArrayLike, abs_scaling: str = "h-1", colorspace: str = "CAM16"):
    # A number of scalings f that map the magnitude [0, infty] to [0, 1] are possible.
    # One desirable property is
    # (1)  f(1/r) = 1 - f(r).
    # This makes sure that the representation of the inverse of a function is exactly as
    # light as the original function is dark. The function g_a(r) = 1 - a^r (with some
    # 0 < a < 1), as it is sometimes suggested (e.g., on Wikipedia
    # <https://en.wikipedia.org/wiki/Domain_coloring>) does _not_ fulfill (1).
    #
    # Here, we are using the simple
    #
    #   h(r) = r^a / (r^a + 1)
    #
    # with a configurable parameter a.
    #  * For a=1.21268891, this function is very close to the popular alternative
    #    2/pi * arctan(r) (which also fulfills the above property)
    #  * For a=1.21428616 is is close to g_{1/2} (between 0 and 1).
    #  * For a=1.49486991 it is close to x/2 (between 0 and 1).
    #
    # Disadvantage of this choice:
    #
    #  h'(r) = (a * r^{a-1} * (r^a + 1) - r^a * a * r^{a-1}) / (r^a + 1) ** 2
    #        = a * r^{a-1} / (r^a + 1) ** 2
    #
    # so h'(r)=0 at r=0 for all a > 1. This means that h(r) has an inflection point in
    # (0, 1) for all a > 1. For 0 < alpha < 1, the derivative at 0 is infty.
    # The arctan version does _not_ have such properties.
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
    #             / 0             for 0 <= x <= 1,
    #   f''(r) = |
    #             \ -1 / r^3   for x > 1,
    #
    #
    #
    # TODO find parametrized function that is free of inflection points for the param=0
    # (or infty) is this last f(r).

    angle = np.arctan2(z.imag, z.real)
    absval_scaled = scale01(np.abs(z), abs_scaling)

    # We may have NaNs, so don't be too strict here.
    # assert np.all(absval_scaled >= 0)
    # assert np.all(absval_scaled <= 1)

    # It'd be lovely if one could claim that the grayscale of the cplot represents
    # exactly the absolute value of the complex number. The grayscale is computed as the
    # Y component of the XYZ-representation of the color, for linear SRGB values as
    #
    #     0.2126 * r + 0.7152 * g + 0.722 * b.
    #
    # Unfortunately, there is no perceptually uniform color space yet that uses
    # Y-luminance. CIELAB, CIECAM02, and CAM16 have their own values.
    if colorspace.upper() == "CAM16":
        cam = colorio.cs.CAM16UCS(0.69, 20, L_A=64 / np.pi / 5)
        srgb = colorio.cs.SrgbLinear()
        # The max radius for which all colors are representable as SRGB is about 21.7.
        # Crank up the colors a little bit to make the images more saturated. This leads
        # to SRGB-cut-off of course.
        # r0 = find_max_srgb_radius(cam, srgb, L=50)
        # r0 = 21.65824845433235
        r0 = 25.0
        # Rotate the angles such that a "green" color represents positive real values.
        # The rotation is chosen such that the ratio g/(r+b) (in rgb) is the largest for
        # the point 1.0.
        offset = 0.916_708 * np.pi
        # Map (r, angle) to a point in the color space; bicone mapping similar to what
        # HSL looks like <https://en.wikipedia.org/wiki/HSL_and_HSV>.
        rd = r0 - r0 * 2 * abs(absval_scaled - 0.5)
        cam_pts = np.array(
            [
                100 * absval_scaled,
                rd * np.cos(angle + offset),
                rd * np.sin(angle + offset),
            ]
        )
        # now just translate to srgb
        srgb_vals = srgb.to_rgb1(srgb.from_xyz100(cam.to_xyz100(cam_pts)))
        srgb_vals *= 1.1
        # Cut off the outliers. This restriction makes the representation less perfect,
        # but that's what it is with the SRGB color space.
        srgb_vals[srgb_vals > 1] = 1.0
        srgb_vals[srgb_vals < 0] = 0.0
    elif colorspace.upper() == "CIELAB":
        cielab = colorio.cs.CIELAB()
        srgb = colorio.cs.SrgbLinear()
        # The max radius is about 29.5, but crank up colors a little bit to make the
        # images more saturated. This leads to SRGB-cut-off of course.
        # r0 = find_max_srgb_radius(cielab, srgb, L=50)
        # r0 = 29.488203674554825
        r0 = 45.0
        # Rotate the angles such that a "green" color represents positive real values.
        # The rotation is chosen such that the ratio g/(r+b) (in rgb) is the largest for
        # the point 1.0.
        offset = 0.893_686_8 * np.pi
        # Map (r, angle) to a point in the color space; bicone mapping similar to what
        # HSL looks like <https://en.wikipedia.org/wiki/HSL_and_HSV>.
        rd = r0 - r0 * 2 * abs(absval_scaled - 0.5)
        lab_pts = np.array(
            [
                100 * absval_scaled,
                rd * np.cos(angle + offset),
                rd * np.sin(angle + offset),
            ]
        )
        # now just translate to srgb
        srgb_vals = srgb.to_rgb1(srgb.from_xyz100(cielab.to_xyz100(lab_pts)))
        # Cut off the outliers. This restriction makes the representation less perfect,
        # but that's what it is with the SRGB color space.
        srgb_vals[srgb_vals > 1] = 1.0
        srgb_vals[srgb_vals < 0] = 0.0
    elif colorspace.upper() == "OKLAB":
        oklab = colorio.cs.OKLAB()
        srgb = colorio.cs.SrgbLinear()
        # The max radius is about 29.5, but crank up colors a little bit to make the
        # images more saturated. This leads to SRGB-cut-off of course.
        # OKLAB is designed such that the D65 whitepoint, scaled to Y=100, has lightness
        # 1. Without the scaling in Y, this results in a whitepoint lightness of
        # 4.64158795.
        # lab = oklab.from_xyz100(colorio.cs.illuminants.whitepoints_cie1931["D65"])
        max_lightness = 4.64158795
        # from .create import find_max_srgb_radius
        # r0 = find_max_srgb_radius(oklab, srgb, L=max_lightness / 2)
        # The actual value is 0.3945164382457733, but allow a slight oversaturation.
        r0 = 0.50
        # Rotate the angles such a "green" color represents positive real values. The
        # rotation is chosen such that the ratio g/(r+b) (in rgb) is the largest for the
        # point 1.0.
        offset = 0.893_686_8 * np.pi
        # Map (r, angle) to a point in the color space; bicone mapping similar to what
        # HSL looks like <https://en.wikipedia.org/wiki/HSL_and_HSV>.
        rd = r0 - r0 * 2 * abs(absval_scaled - 0.5)
        lab_pts = np.array(
            [
                max_lightness * absval_scaled,
                rd * np.cos(angle + offset),
                rd * np.sin(angle + offset),
            ]
        )
        # now just translate to srgb
        srgb_vals = srgb.to_rgb1(srgb.from_xyz100(oklab.to_xyz100(lab_pts)))
        # Cut off the outliers. This restriction makes the representation less perfect,
        # but that's what it is with the SRGB color space.
        srgb_vals[srgb_vals > 1] = 1.0
        srgb_vals[srgb_vals < 0] = 0.0
    else:
        assert (
            colorspace.upper() == "HSL"
        ), f"Illegal colorspace {colorspace}. Pick one of CAM16, CIELAB, HSL."
        hsl = colorio.cs.HSL()
        # rotate by 120 degrees to have green (0, 1, 0) for real positive numbers
        offset = 120
        hsl_vals = np.array(
            [
                np.mod(angle / (2 * np.pi) * 360 + offset, 360),
                np.ones(angle.shape),
                absval_scaled,
            ]
        )
        srgb_vals = hsl.to_rgb1(hsl_vals)
        # iron out the -1.82131e-17 round-offs
        srgb_vals[srgb_vals < 0] = 0

    return np.moveaxis(srgb_vals, 0, -1)


def tripcolor(triang, z, abs_scaling: str = "h-1"):
    rgb = get_srgb1(z, abs_scaling=abs_scaling)

    # https://github.com/matplotlib/matplotlib/issues/10265#issuecomment-358684592
    n = z.shape[0]
    z2 = np.arange(n)
    cmap = matplotlib.colors.LinearSegmentedColormap.from_list("mymap", rgb, N=n)
    plt.tripcolor(triang, z2, shading="gouraud", cmap=cmap)
