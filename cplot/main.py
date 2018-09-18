# -*- coding: utf-8 -*-
#
from __future__ import division

import matplotlib.pyplot as plt
import numpy

import colorio


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


def show(*args, **kwargs):
    plot(*args, **kwargs)
    plt.show()
    return


def savefig(filename, *args, **kwargs):
    plot(*args, **kwargs)
    plt.savefig(filename, transparent=True, bbox_inches="tight")
    return


def get_variant_a(alpha):
    def scaler(r):
        #          ( alpha * r  if r < 1,
        #   f(r) = {
        #          ( gamma * (1 - 1/(r-beta)) + delta  otherwise.
        #
        # with
        #
        #   beta = (2*alpha - 1) / alpha
        #   gamma = (1 - beta)**2 * alpha
        #   delta = 1 - gamma
        #
        # The parameters are chosen such that the function is continuously
        # differentiable and f(x) -> 1 for x -> infty. This choice of f has the
        # advantage of being linear between 0 and 1 and can be scaled for lightness with
        # alpha.
        absval_scaled = r.copy()
        is_smaller = absval_scaled < 1
        alpha = 0.5
        absval_scaled[is_smaller] = alpha * absval_scaled[is_smaller]

        beta = (2 * alpha - 1) / alpha
        gamma = (1 - beta)**2 * alpha
        delta = 1 - gamma
        absval_scaled[~is_smaller] = gamma * (1 - 1 / (absval_scaled[~is_smaller] - beta)) + delta
        #
        # absval_scaled = absval / (absval + 1)
        #
        # absval_scaled = 2/numpy.pi * numpy.arctan(absval)
        return absval_scaled

    return scaler


def get_srgb(angle, absval_scaled):
    assert numpy.all(absval_scaled >= 0)
    assert numpy.all(absval_scaled <= 1)

    variant = "CAM16UCS"
    if variant == "CAM16UCS":
        L_A = 64 / numpy.pi / 5
        cam = colorio.CAM16UCS(0.69, 20, L_A)
        srgb = colorio.SrgbLinear()
        # r0 = find_max_srgb_radius(cam, srgb, L=50)
        r0 = 21.65824845433235
    else:
        assert False

    # map (r, angle) to a point in the color space
    rd = r0 - r0 * 2 * abs(absval_scaled - 0.5)
    cam_pts = numpy.array(
        [
            100 * absval_scaled,
            rd * numpy.cos(angle + 0.7 * numpy.pi),
            rd * numpy.sin(angle + 0.7 * numpy.pi),
        ]
    )

    # now just translate to srgb and plot the image
    srgb_vals = srgb.to_srgb1(srgb.from_xyz100(cam.to_xyz100(cam_pts)))
    # assert numpy.all(srgb.from_xyz100(cam.to_xyz100(cam_pts)) <= 1.0)
    srgb_vals[srgb_vals > 1] = 1.0
    srgb_vals[srgb_vals < 0] = 0.0

    return numpy.moveaxis(srgb_vals, 0, -1)


def plot(f, xmin, xmax, ymin, ymax, nx, ny, abs_scaling=get_variant_a(0.5)):
    assert xmax > xmin
    assert ymax > ymin
    hx = (xmax - xmin) / nx
    x = numpy.linspace(xmin + hx / 2, xmax - hx / 2, nx)
    hy = (ymax - ymin) / ny
    y = numpy.linspace(ymin + hy / 2, ymax - hy / 2, ny)

    X = numpy.meshgrid(x, y)
    val = f(X[0] + 1j * X[1])

    angle = numpy.arctan2(val.imag, val.real)
    absval_scaled = abs_scaling(numpy.abs(val))
    srgb_vals = get_srgb(angle, absval_scaled)

    plt.imshow(
        srgb_vals,
        extent=(x.min(), x.max(), y.max(), y.min()),
        interpolation="nearest",
        origin="lower",
        aspect="equal",
    )
    return
