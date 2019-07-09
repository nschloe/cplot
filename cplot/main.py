import matplotlib
import matplotlib.pyplot as plt
import numpy

import colorio


def show(*args, **kwargs):
    plot(*args, **kwargs)
    plt.show()
    return


def savefig(filename, *args, **kwargs):
    plot(*args, **kwargs)
    plt.savefig(filename, transparent=True, bbox_inches="tight")
    return


def get_srgb1(z, alpha=1):
    assert alpha > 0

    # Other possible scalings:
    # def abs_scaling(r):
    #     # Fulfills f(1/r) = 1 - f(r).
    #    return 2 / numpy.pi * numpy.arctan(r)

    def abs_scaling(r):
        # Fulfills f(1/r) = 1 - f(r) for any alpha > 0
        return r ** alpha / (r ** alpha + 1)

    angle = numpy.arctan2(z.imag, z.real)
    absval_scaled = abs_scaling(numpy.abs(z))

    assert numpy.all(absval_scaled >= 0)
    assert numpy.all(absval_scaled <= 1)

    # It'd be lovely if one could claim that the grayscale of the cplot represents
    # exactly the absolute value of the complex number. The grayscale is computed as the
    # Y component of the XYZ-representation of the color, for linear SRGB values as
    #
    #     0.2126 * r + 0.7152 * g + 0.722 * b.
    #
    # Unfortunately, there is no perceptually uniform color space yet that uses
    # Y-luminance. CIELAB, CIECAM02, and CAM16 have their own values.
    L_A = 64 / numpy.pi / 5
    cam = colorio.CAM16UCS(0.69, 20, L_A)
    srgb = colorio.SrgbLinear()

    # The max radius is about 21.7, but crank up colors a little bit to make the images
    # more saturated. This leads to SRGB-cut-off of course.
    # r0 = find_max_srgb_radius(cam, srgb, L=50)
    # r0 = 21.65824845433235
    r0 = 25.0

    # map (r, angle) to a point in the color space
    rd = r0 - r0 * 2 * abs(absval_scaled - 0.5)

    # Rotate the angles such a "green" color represents positive real values. The
    # rotation is chosen such that the ratio g/(r+b) (in rgb) is the largest for the
    # point 1.0.
    offset = 0.916708 * numpy.pi
    cam_pts = numpy.array(
        [
            100 * absval_scaled,
            rd * numpy.cos(angle + offset),
            rd * numpy.sin(angle + offset),
        ]
    )

    # now just translate to srgb
    srgb_vals = srgb.to_srgb1(srgb.from_xyz100(cam.to_xyz100(cam_pts)))
    # Cut off the outliers. This restriction makes the representation less perfect, but
    # that's what it is with the SRGB color space.
    srgb_vals[srgb_vals > 1] = 1.0
    srgb_vals[srgb_vals < 0] = 0.0

    return numpy.moveaxis(srgb_vals, 0, -1)


def plot(f, xmin, xmax, ymin, ymax, nx, ny, alpha=1):
    assert xmax > xmin
    assert ymax > ymin
    hx = (xmax - xmin) / nx
    x = numpy.linspace(xmin + hx / 2, xmax - hx / 2, nx)
    hy = (ymax - ymin) / ny
    y = numpy.linspace(ymin + hy / 2, ymax - hy / 2, ny)

    X = numpy.meshgrid(x, y)

    z = X[0] + 1j * X[1]

    srgb_vals = get_srgb1(f(z), alpha=alpha)

    plt.imshow(
        srgb_vals,
        extent=(x.min(), x.max(), y.max(), y.min()),
        interpolation="nearest",
        origin="lower",
        aspect="equal",
    )
    return


def tripcolor(triang, z, alpha=1):
    rgb = get_srgb1(z, alpha=alpha)

    # https://github.com/matplotlib/matplotlib/issues/10265#issuecomment-358684592
    n = z.shape[0]
    z2 = numpy.arange(n)
    cmap = matplotlib.colors.LinearSegmentedColormap.from_list("mymap", rgb, N=n)
    plt.tripcolor(triang, z2, shading="gouraud", cmap=cmap)
    return
