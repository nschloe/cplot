# -*- coding: utf-8 -*-
#
import matplotlib
import numpy
import scipy.special
import mpmath

import cplot


def test_create():
    rgb = cplot.create_colormap(L=50)
    cmap = matplotlib.colors.LinearSegmentedColormap.from_list(
        "custom", rgb.T, N=len(rgb.T)
    )
    # cmap = 'gray'

    cplot.show_linear(rgb)
    cplot.show_circular(rgb, rot=-numpy.pi * 4 / 12)
    # cplot.show_circular(rgb, rot=-numpy.pi * 18/12)
    cplot.show_kovesi_test_image(cmap)
    return


def zeta(z):
    vals = [[mpmath.zeta(val) for val in row] for row in z]
    out = numpy.array(
        [[float(val.real) + 1j * float(val.imag) for val in row] for row in vals]
    )
    return out


def test_show():
    cplot.savefig("z1.png", lambda z: z ** 1, -2, +2, -2, +2, 101, 101)
    cplot.savefig("z2.png", lambda z: z ** 2, -2, +2, -2, +2, 101, 101)
    cplot.savefig("z3.png", lambda z: z ** 3, -2, +2, -2, +2, 101, 101)

    cplot.savefig("1z.png", lambda z: 1 / z, -2, +2, -2, +2, 100, 100)
    cplot.savefig("z-absz.png", lambda z: z / abs(z), -2, +2, -2, +2, 100, 100)
    cplot.savefig("z+1-z-1.png", lambda z: (z + 1) / (z - 1), -5, +5, -5, +5, 101, 101)

    cplot.savefig("sqrt.png", numpy.sqrt, -5, +5, -5, +5, 200, 200)
    cplot.savefig("log.png", numpy.log, -5, +5, -5, +5, 200, 200)
    cplot.savefig("exp.png", numpy.exp, -5, +5, -5, +5, 200, 200)

    cplot.savefig("sin.png", numpy.sin, -5, +5, -5, +5, 200, 200)
    cplot.savefig("cos.png", numpy.cos, -5, +5, -5, +5, 200, 200)
    cplot.savefig("tan.png", numpy.tan, -5, +5, -5, +5, 200, 200)

    cplot.savefig("gamma.png", scipy.special.gamma, -5, +5, -5, +5, 200, 200)
    cplot.savefig("digamma.png", scipy.special.digamma, -5, +5, -5, +5, 200, 200)
    cplot.savefig("zeta.png", zeta, -30, +30, -30, +30, 200, 200)

    # a = 10
    # cplot.savefig("bessel0.png", lambda z: scipy.special.jv(0, z), -a, +a, -a, +a, 100, 100)
    # cplot.savefig("bessel1.png", lambda z: scipy.special.jv(1, z), -a, +a, -a, +a, 100, 100)
    # cplot.savefig("bessel2.png", lambda z: scipy.special.jv(2, z), -a, +a, -a, +a, 100, 100)
    # cplot.savefig("bessel3.png", lambda z: scipy.special.jv(3, z), -a, +a, -a, +a, 100, 100)
    return


def scaler_arctan(r):
    # Fulfills f(1/r) = 1 - f(r).
    return 2 / numpy.pi * numpy.arctan(r)


def scaler_fraction(r):
    # Fulfills f(1/r) = 1 - f(r).
    # any alpha > 0 is good
    alpha = 1.0
    return r ** alpha / (r ** alpha + 1)


if __name__ == "__main__":
    # test_create()
    test_show()
