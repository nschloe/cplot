# -*- coding: utf-8 -*-
#
from __future__ import division

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
        gamma = (1 - beta) ** 2 * alpha
        delta = 1 - gamma
        absval_scaled[~is_smaller] = (
            gamma * (1 - 1 / (absval_scaled[~is_smaller] - beta)) + delta
        )
        #
        # absval_scaled = absval / (absval + 1)
        #
        # absval_scaled = 2/numpy.pi * numpy.arctan(absval)
        return absval_scaled

    return scaler


def get_srgb(angle, absval_scaled):
    assert numpy.all(absval_scaled >= 0)
    assert numpy.all(absval_scaled <= 1)

    # assert variant == "CAM16UCS":
    L_A = 64 / numpy.pi / 5
    cam = colorio.CAM16UCS(0.69, 20, L_A)
    srgb = colorio.SrgbLinear()
    # r0 = find_max_srgb_radius(cam, srgb, L=50)
    r0 = 21.65824845433235

    # map (r, angle) to a point in the color space
    rd = r0 - r0 * 2 * abs(absval_scaled - 0.5)

    # rotate the angles such a "green" color represents positive real values
    offset = 1.0 * numpy.pi
    cam_pts = numpy.array(
        [
            100 * absval_scaled,
            rd * numpy.cos(angle + offset),
            rd * numpy.sin(angle + offset),
        ]
    )

    # now just translate to srgb
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
