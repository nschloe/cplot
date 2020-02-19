import matplotlib
import matplotlib.pyplot as plt
import numpy

import colorio


def get_srgb1(z, alpha=1, colorspace="CAM16"):
    assert alpha >= 0
    # A number of scalings f that map the magnitude [0, infty] to [0, 1] are possible.
    # One desirable property is
    # (1)  f(1/r) = 1 - f(r).
    # This makes sure that the representation of the inverse of a function is exactly as
    # light as the original function is dark. The function g_a(r) = 1 - a^|r| (with some
    # 0 < a < 1), as it is sometimes suggested (e.g., on Wikipedia
    # <https://en.wikipedia.org/wiki/Domain_coloring>) does _not_ fulfill (1).  The
    # function 2/pi * arctan(r) is _very_ close to g_(1/2) between 0 and 1 and has that
    # property, so this is good alternative. Here, we are using the simple r^a / r^a+1
    # with a configurable parameter a.

    def abs_scaling(r):
        # Fulfills (1) for any alpha >= 0
        return r ** alpha / (r ** alpha + 1)

    # def abs_scaling(r):
    #     # Fulfills (1).
    #    return 2 / numpy.pi * numpy.arctan(r)

    angle = numpy.arctan2(z.imag, z.real)
    absval_scaled = abs_scaling(numpy.abs(z))

    # We may have NaNs, so don't be too strict here.
    # assert numpy.all(absval_scaled >= 0)
    # assert numpy.all(absval_scaled <= 1)

    # It'd be lovely if one could claim that the grayscale of the cplot represents
    # exactly the absolute value of the complex number. The grayscale is computed as the
    # Y component of the XYZ-representation of the color, for linear SRGB values as
    #
    #     0.2126 * r + 0.7152 * g + 0.722 * b.
    #
    # Unfortunately, there is no perceptually uniform color space yet that uses
    # Y-luminance. CIELAB, CIECAM02, and CAM16 have their own values.
    if colorspace.upper() == "CAM16":
        L_A = 64 / numpy.pi / 5
        cam = colorio.CAM16UCS(0.69, 20, L_A)
        srgb = colorio.SrgbLinear()
        # The max radius is about 21.7, but crank up colors a little bit to make the
        # images more saturated. This leads to SRGB-cut-off of course.
        # r0 = find_max_srgb_radius(cam, srgb, L=50)
        # r0 = 21.65824845433235
        r0 = 25.0
        # Rotate the angles such a "green" color represents positive real values. The
        # rotation is chosen such that the ratio g/(r+b) (in rgb) is the largest for the
        # point 1.0.
        offset = 0.916_708 * numpy.pi
        # Map (r, angle) to a point in the color space; bicone mapping similar to what
        # HSL looks like <https://en.wikipedia.org/wiki/HSL_and_HSV>.
        rd = r0 - r0 * 2 * abs(absval_scaled - 0.5)
        cam_pts = numpy.array(
            [
                100 * absval_scaled,
                rd * numpy.cos(angle + offset),
                rd * numpy.sin(angle + offset),
            ]
        )
        # now just translate to srgb
        srgb_vals = srgb.to_srgb1(srgb.from_xyz100(cam.to_xyz100(cam_pts)))
        # Cut off the outliers. This restriction makes the representation less perfect,
        # but that's what it is with the SRGB color space.
        srgb_vals[srgb_vals > 1] = 1.0
        srgb_vals[srgb_vals < 0] = 0.0
    elif colorspace.upper() == "CIELAB":
        cielab = colorio.CIELAB()
        srgb = colorio.SrgbLinear()
        # The max radius is about 29.5, but crank up colors a little bit to make the
        # images more saturated. This leads to SRGB-cut-off of course.
        # r0 = find_max_srgb_radius(cielab, srgb, L=50)
        # r0 = 29.488203674554825
        r0 = 45.0
        # Rotate the angles such a "green" color represents positive real values. The
        # rotation is chosen such that the ratio g/(r+b) (in rgb) is the largest for the
        # point 1.0.
        offset = 0.893_686_8 * numpy.pi
        # Map (r, angle) to a point in the color space; bicone mapping similar to what
        # HSL looks like <https://en.wikipedia.org/wiki/HSL_and_HSV>.
        rd = r0 - r0 * 2 * abs(absval_scaled - 0.5)
        lab_pts = numpy.array(
            [
                100 * absval_scaled,
                rd * numpy.cos(angle + offset),
                rd * numpy.sin(angle + offset),
            ]
        )
        # now just translate to srgb
        srgb_vals = srgb.to_srgb1(srgb.from_xyz100(cielab.to_xyz100(lab_pts)))
        # Cut off the outliers. This restriction makes the representation less perfect,
        # but that's what it is with the SRGB color space.
        srgb_vals[srgb_vals > 1] = 1.0
        srgb_vals[srgb_vals < 0] = 0.0
    else:
        assert (
            colorspace.upper() == "HSL"
        ), f"Illegal colorspace {colorspace}. Pick one of CAM16, CIELAB, HSL."
        hsl = colorio.HSL()
        # rotate by 120 degrees to have green (0, 1, 0) for real positive numbers
        offset = 120
        hsl_vals = numpy.array(
            [
                numpy.mod(angle / (2 * numpy.pi) * 360 + offset, 360),
                numpy.ones(angle.shape),
                absval_scaled,
            ]
        )
        srgb_vals = hsl.to_srgb1(hsl_vals)
        # iron out the -1.82131e-17 round-offs
        srgb_vals[srgb_vals < 0] = 0

    return numpy.moveaxis(srgb_vals, 0, -1)


def plot(*args, **kwargs):
    vals, extent = _get_srgb_vals(*args, **kwargs)
    plt.imshow(
        vals, extent=extent, interpolation="nearest", origin="lower", aspect="equal"
    )
    return


def show(*args, **kwargs):
    plot(*args, **kwargs)
    plt.show()


def save_fig(filename, *args, **kwargs):
    plot(*args, **kwargs)
    plt.savefig(filename, transparent=True, bbox_inches="tight")


def save_img(filename, *args, **kwargs):
    vals, _ = _get_srgb_vals(*args, **kwargs)
    matplotlib.image.imsave(filename, vals)


def _get_srgb_vals(f, xmin, xmax, ymin, ymax, nx, ny, alpha=1, colorspace="cam16"):
    assert xmax > xmin
    assert ymax > ymin
    hx = (xmax - xmin) / nx
    x = numpy.linspace(xmin + hx / 2, xmax - hx / 2, nx)
    hy = (ymax - ymin) / ny
    y = numpy.linspace(ymin + hy / 2, ymax - hy / 2, ny)

    X = numpy.meshgrid(x, y)

    z = X[0] + 1j * X[1]

    return (
        get_srgb1(f(z), alpha=alpha, colorspace=colorspace),
        (x.min(), x.max(), y.min(), y.max()),
    )


def tripcolor(triang, z, alpha=1):
    rgb = get_srgb1(z, alpha=alpha)

    # https://github.com/matplotlib/matplotlib/issues/10265#issuecomment-358684592
    n = z.shape[0]
    z2 = numpy.arange(n)
    cmap = matplotlib.colors.LinearSegmentedColormap.from_list("mymap", rgb, N=n)
    plt.tripcolor(triang, z2, shading="gouraud", cmap=cmap)
