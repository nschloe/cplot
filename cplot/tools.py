# -*- coding: utf-8 -*-
#
from __future__ import division

import matplotlib.pyplot as plt
import numpy

import colorio


def show_kovesi(cmap):
    '''Visual color map test after Peter Kovesi
    <https://arxiv.org/abs/1509.03700>.
    '''
    n = 300
    x = numpy.arange(n+1)/n
    y = numpy.arange(n+1)/n / 3
    X, Y = numpy.meshgrid(x, y)
    # From <https://arxiv.org/abs/1509.03700>:
    # It consists of a sine wave superimposed on a ramp function, this provides
    # a set of constant magnitude features presented at different offsets. The
    # spatial frequency of the sine wave is chosen to lie in the range at which
    # the human eye is most sensitive, and its amplitude is set so that the
    # range from peak to trough represents a series of features that are 10% of
    # the total data range. The amplitude of the sine wave is modulated from
    # its full value at the top of the image to zero at the bottom.
    z = X + (3*Y)**2 * 0.05 * numpy.sin(100*numpy.pi*X)
    # z = X + 0.05 * numpy.sin(100*numpy.pi*X*Y)

    plt.imshow(
        z, extent=(x.min(), x.max(), y.max(), y.min()),
        interpolation='nearest', cmap=cmap,
        origin='lower',
        aspect='equal'
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

    alpha = numpy.mod(numpy.arctan2(y, x)-rot, 2*numpy.pi)

    m = vals.shape[1]
    ls = numpy.linspace(0, 2*numpy.pi, m, endpoint=False)
    r = numpy.interp(alpha.reshape(-1), ls, vals[0]).reshape(alpha.shape)
    g = numpy.interp(alpha.reshape(-1), ls, vals[1]).reshape(alpha.shape)
    b = numpy.interp(alpha.reshape(-1), ls, vals[2]).reshape(alpha.shape)
    out = numpy.array([r, g, b])

    plt.imshow(out.T)
    plt.show()
    return


def find_max_srgb_radius(cs, srgb, L=50):
    # Go into the CAM16-UCS color space and find the circle in the L=50-plane
    # with the center (50, 0, 0) such that it's as large as possible while
    # still being in the SRGB gamut.
    n = 256
    alpha = numpy.linspace(0, 2*numpy.pi, n, endpoint=False)

    # bisection
    r0 = 0.0
    r1 = 100.0
    tol = 1.0e-6
    while r1 - r0 > tol:
        r = 0.5 * (r1 + r0)

        pts = numpy.array([
            numpy.full(n, L),
            r * numpy.cos(alpha),
            r * numpy.sin(alpha),
            ])
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
    alpha = numpy.linspace(0, 2*numpy.pi, n, endpoint=False)

    pts = numpy.array([
        numpy.full(n, L),
        r0 * numpy.cos(alpha),
        r0 * numpy.sin(alpha),
        ])
    vals = srgb.from_xyz100(cam.to_xyz100(pts))

    # show the colors
    vals = srgb.to_srgb1(vals)
    return vals


def show(*args, **kwargs):
    plot(*args, **kwargs)
    plt.show()
    return


def plot(f, xmin, xmax, ymin, ymax, nx, ny):
    assert xmax > xmin
    assert ymax > ymin
    hx = (xmax - xmin) / nx
    x = numpy.linspace(xmin+hx/2, xmax-hx/2, nx)
    hy = (ymax - ymin) / ny
    y = numpy.linspace(ymin+hy/2, ymax-hy/2, ny)

    X = numpy.meshgrid(x, y)
    val = f(X[0] + 1j*X[1])

    angle = numpy.arctan2(val.imag, val.real)
    absval = numpy.abs(val)

    # Map |f(z)| to [0, 1] such that f(1/z) = 1-f(z). There are many possibilities
    # for doing so, e.g.,
    #
    #   2/pi * arctan(z)
    #   z^alpha / (z^alpha+1) with any alpha > 0.
    #
    # Pick
    #          ( z/2  if z < 1,
    #   f(z) = {
    #          ( 1 - 1/2/z  otherwise.
    #
    # This function has the advantage of not intrucing distortion between 0 and
    # 1.
    absval_scaled = absval.copy()
    is_smaller = absval_scaled < 1
    absval_scaled[is_smaller] = absval_scaled[is_smaller] / 2
    absval_scaled[~is_smaller] = 1 - 0.5/absval_scaled[~is_smaller]
    #
    # absval_scaled = absval / (absval + 1)
    #
    # absval_scaled = 2/numpy.pi * numpy.arctan(absval)

    assert numpy.all(absval_scaled >= 0)
    assert numpy.all(absval_scaled <= 1)

    # TODO hardcode radius
    L_A = 64 / numpy.pi / 5
    cam = colorio.CAM16UCS(0.69, 20, L_A)
    srgb = colorio.SrgbLinear()
    r0 = find_max_srgb_radius(cam, srgb, L=50)
    # r0 = 30.0

    # map (r, angle) to a point in the color space
    rd = r0 - r0 * 2 * abs(absval_scaled - 0.5)
    cam_pts = numpy.array([
        100 * absval_scaled,
        rd * numpy.cos(angle + 0.7 * numpy.pi),
        rd * numpy.sin(angle + 0.7 * numpy.pi),
        ])

    # now just translate to srgb and plot the image
    srgb_vals = srgb.to_srgb1(srgb.from_xyz100(cam.to_xyz100(cam_pts)))
    # assert numpy.all(srgb.from_xyz100(cam.to_xyz100(cam_pts)) <= 1.0)
    print(numpy.any(numpy.isnan(srgb_vals)))
    srgb_vals[srgb_vals > 1] = 1.0
    srgb_vals[srgb_vals < 0] = 0.0

    plt.imshow(
        numpy.moveaxis(srgb_vals, 0, -1),
        extent=(x.min(), x.max(), y.max(), y.min()),
        interpolation='nearest',
        origin='lower',
        aspect='equal',
        )
    return
