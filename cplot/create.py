# -*- coding: utf-8 -*-
#
import colorio
import matplotlib.pyplot as plt
import numpy


def show_kovesi_test_image(cmap):
    """Visual color map test after Peter Kovesi
    <https://arxiv.org/abs/1509.03700>.
    """
    n = 300
    x = numpy.arange(n + 1) / n
    y = numpy.arange(n + 1) / n / 3
    X, Y = numpy.meshgrid(x, y)
    # From <https://arxiv.org/abs/1509.03700>:
    # It consists of a sine wave superimposed on a ramp function, this provides a set of
    # constant magnitude features presented at different offsets. The spatial frequency
    # of the sine wave is chosen to lie in the range at which the human eye is most
    # sensitive, and its amplitude is set so that the range from peak to trough
    # represents a series of features that are 10% of the total data range. The
    # amplitude of the sine wave is modulated from its full value at the top of the
    # image to zero at the bottom.
    z = X + (3 * Y) ** 2 * 0.05 * numpy.sin(100 * numpy.pi * X)
    # z = X + 0.05 * numpy.sin(100*numpy.pi*X*Y)

    plt.imshow(
        z,
        extent=(x.min(), x.max(), y.max(), y.min()),
        interpolation="nearest",
        cmap=cmap,
        origin="lower",
        aspect="equal",
    )

    plt.xticks([])
    plt.yticks([])

    plt.show()
    return


def show_linear(vals):
    plt.imshow(numpy.multiply.outer(numpy.ones(60), vals.T))
    plt.show()
    return


def show_circular(vals, rot=0.0):
    n = 256
    x, y = numpy.meshgrid(numpy.linspace(-n, +n), numpy.linspace(-n, +n))

    alpha = numpy.mod(numpy.arctan2(y, x) - rot, 2 * numpy.pi)

    m = vals.shape[1]
    ls = numpy.linspace(0, 2 * numpy.pi, m, endpoint=False)
    r = numpy.interp(alpha.reshape(-1), ls, vals[0]).reshape(alpha.shape)
    g = numpy.interp(alpha.reshape(-1), ls, vals[1]).reshape(alpha.shape)
    b = numpy.interp(alpha.reshape(-1), ls, vals[2]).reshape(alpha.shape)
    out = numpy.array([r, g, b])

    plt.imshow(out.T)
    plt.show()
    return


def find_max_srgb_radius(cs, srgb, L=50):
    # Go into the CAM16-UCS color space and find the circle in the L=50-plane with the
    # center (50, 0, 0) such that it's as large as possible while still being in the
    # SRGB gamut.
    n = 256
    alpha = numpy.linspace(0, 2 * numpy.pi, n, endpoint=False)

    # bisection
    r0 = 0.0
    r1 = 100.0
    tol = 1.0e-6
    while r1 - r0 > tol:
        r = 0.5 * (r1 + r0)

        pts = numpy.array(
            [numpy.full(n, L), r * numpy.cos(alpha), r * numpy.sin(alpha)]
        )
        vals = srgb.from_xyz100(cs.to_xyz100(pts))

        if numpy.any(vals < 0) or numpy.any(vals > 1):
            r1 = r
        else:
            r0 = r
    return r0


def create_colormap(L=50):
    L_A = 64 / numpy.pi / 5
    cam = colorio.CAM16UCS(0.69, 20, L_A)
    # cam = colorio.CAM02('UCS', 0.69, 20, L_A)
    # cam = colorio.CIELAB()
    srgb = colorio.SrgbLinear()

    r0 = find_max_srgb_radius(cam, srgb, L=L)

    n = 256
    alpha = numpy.linspace(0, 2 * numpy.pi, n, endpoint=False)

    pts = numpy.array([numpy.full(n, L), r0 * numpy.cos(alpha), r0 * numpy.sin(alpha)])
    vals = srgb.from_xyz100(cam.to_xyz100(pts))

    # show the colors
    vals = srgb.to_srgb1(vals)
    return vals
