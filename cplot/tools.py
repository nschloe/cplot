# -*- coding: utf-8 -*-
#
from __future__ import division

import colorio
import matplotlib.pyplot as plt
import numpy


def colormap_test_function(X):
    '''Modeled after Peter Kovesi <https://arxiv.org/abs/1509.03700>.
    '''
    return X[0] + X[1] * 0.1 * numpy.sin(50*numpy.pi*X[0])


def colormap_test(colormap):
    return


def show_linear(vals):
    plt.imshow(numpy.multiply.outer(numpy.ones(60), vals.T))
    plt.show()
    return


def create_colormap():
    # Go into the CAM16-UCS color space and find the circle in the L=50-plane
    # with the center (50, 0, 0) such that it's as large as possible while
    # still being in the SRGB gamut.
    L = 60
    n = 256
    alpha = numpy.arange(n) / n * 2*numpy.pi

    L_A = 64 / numpy.pi / 5
    cam = colorio.CAM16UCS(0.69, 20, L_A)
    # cam = colorio.CAM02('UCS', 0.69, 20, L_A)
    # cam = colorio.CIELUV()
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
