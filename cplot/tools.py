# -*- coding: utf-8 -*-
#
from __future__ import division

import colorio
import matplotlib
import matplotlib.pyplot as plt
import numpy


def show_kovesi(rgb):
    '''Visual color map test after Peter Kovesi
    <https://arxiv.org/abs/1509.03700>.
    '''
    cmap = matplotlib.colors.LinearSegmentedColormap.from_list(
        'custom', rgb.T, N=len(rgb.T)
        )
    cmap = 'viridis'

    n = 300
    x = numpy.arange(n+1)/n
    y = numpy.arange(n+1)/n / 3
    X, Y = numpy.meshgrid(x, y)
    z = X + (3*Y)**2 * 0.05 * numpy.sin(100*numpy.pi*X)

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
    x, y = numpy.meshgrid(numpy.arange(-n, n+1) / n, numpy.arange(-n, n+1) / n)

    alpha = numpy.mod(numpy.arctan2(y, x)-rot, 2*numpy.pi)

    m = vals.shape[1]
    r = numpy.interp(
        alpha.reshape(-1), 2*numpy.pi*numpy.arange(m)/m, vals[0]
        ).reshape(alpha.shape)
    g = numpy.interp(
        alpha.reshape(-1), 2*numpy.pi*numpy.arange(m)/m, vals[1]
        ).reshape(alpha.shape)
    b = numpy.interp(
        alpha.reshape(-1), 2*numpy.pi*numpy.arange(m)/m, vals[2]
        ).reshape(alpha.shape)
    out = numpy.array([r, g, b])

    plt.imshow(out.T)
    plt.show()
    return


def create_colormap(L=50):
    # Go into the CAM16-UCS color space and find the circle in the L=50-plane
    # with the center (50, 0, 0) such that it's as large as possible while
    # still being in the SRGB gamut.
    n = 256
    alpha = numpy.arange(n) / n * 2*numpy.pi

    L_A = 64 / numpy.pi / 5
    cam = colorio.CAM16UCS(0.69, 20, L_A)
    # cam = colorio.CAM02('UCS', 0.69, 20, L_A)
    # cam = colorio.CIELAB()
    srgb = colorio.SrgbLinear()

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
        vals = srgb.from_xyz100(cam.to_xyz100(pts))

        if numpy.any(vals < 0) or numpy.any(vals > 1):
            r1 = r
        else:
            r0 = r

    pts = numpy.array([
        numpy.full(n, L),
        r0 * numpy.cos(alpha),
        r0 * numpy.sin(alpha),
        ])
    vals = srgb.from_xyz100(cam.to_xyz100(pts))

    # show the colors
    vals = srgb.to_srgb1(vals)
    return vals
