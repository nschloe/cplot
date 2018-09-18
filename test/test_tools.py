# -*- coding: utf-8 -*-
#
import matplotlib
import numpy
import scipy.special

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
    cplot.show_kovesi(cmap)
    return


def test_show():
    # cplot.show(lambda z: -1j * z ** 0, -1, +1, -1, +1, 101, 101)

    # cplot.savefig("z2.png", lambda z: z ** 2, -2, +2, -2, +2, 101, 101)
    # cplot.show(lambda z: z ** 2, -2, +2, -2, +2, 101, 101)

    # cplot.show(lambda z: numpy.real(z), -1, +1, -1, +1, 101, 101)
    # cplot.show(lambda z: 1j * numpy.imag(z), -1, +1, -1, +1, 101, 101)

    # cplot.savefig("1z.png", lambda z: 1 / z, -1, +1, -1, +1, 101, 101)
    # cplot.savefig("z+1-z-1.png", lambda z: (z + 1) / (z - 1), -5, +5, -5, +5, 101, 101)
    # cplot.savefig("z-1-z+1.png", lambda z: (z - 1) / (z + 1), -5, +5, -5, +5, 101, 101)

    # cplot.savefig("tan.png", numpy.tan, -5, +5, -5, +5, 200, 200)
    # cplot.savefig("sin.png", numpy.sin, -5, +5, -5, +5, 200, 200)
    # cplot.savefig("cos.png", numpy.cos, -5, +5, -5, +5, 200, 200)

    # cplot.savefig("log.png", numpy.log, -5, +5, -5, +5, 200, 200)
    # cplot.savefig("exp.png", numpy.exp, -5, +5, -5, +5, 200, 200)
    cplot.savefig("gamma.png", scipy.special.gamma, -5, +5, -5, +5, 200, 200)
    return


def scaler_arctan(r):
    # Fulfills f(1/r) = 1 - f(r).
    return 2 / numpy.pi * numpy.arctan(r)


def scaler_fraction(r):
    # Fulfills f(1/r) = 1 - f(r).
    # any alpha > 0 is good
    alpha = 1.0
    return r**alpha / (r**alpha + 1)


if __name__ == "__main__":
    test_show()
