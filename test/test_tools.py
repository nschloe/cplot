import matplotlib
import mpmath
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
    cplot.show_kovesi_test_image(cmap)
    return


def zeta(z):
    vals = [[mpmath.zeta(val) for val in row] for row in z]
    out = numpy.array(
        [[float(val.real) + 1j * float(val.imag) for val in row] for row in vals]
    )
    return out


def test_array():
    numpy.random.seed(0)
    n = 5
    z = numpy.random.rand(n) + 1j * numpy.random.rand(n)
    vals = cplot.get_srgb1(z)
    assert vals.shape == (n, 3)
    return


def test_cam16():
    # cplot.save_fig("z6_1.png", lambda z: z ** 6 - 1, -2, +2, -2, +2, 200, 200)
    # cplot.save_fig(
    #     "f025.png",
    #     lambda z: (z ** 2 - 1) * (z - 2 - 1j) ** 2 / (z ** 2 + 2 + 2j),
    #     -3,
    #     +3,
    #     -3,
    #     +3,
    #     200,
    #     200,
    #     alpha=0.25
    # )

    n = 401
    cplot.save_img("z1.png", lambda z: z ** 1, -2, +2, -2, +2, n, n)
    cplot.save_img("z2.png", lambda z: z ** 2, -2, +2, -2, +2, n, n)
    cplot.save_img("z3.png", lambda z: z ** 3, -2, +2, -2, +2, n, n)

    cplot.save_fig("1z.png", lambda z: 1 / z, -2, +2, -2, +2, 100, 100)
    cplot.save_fig("z-absz.png", lambda z: z / abs(z), -2, +2, -2, +2, 100, 100)
    cplot.save_fig("z+1-z-1.png", lambda z: (z + 1) / (z - 1), -5, +5, -5, +5, 101, 101)

    cplot.save_fig("root2.png", numpy.sqrt, -2, +2, -2, +2, 200, 200)
    cplot.save_fig("root3.png", lambda x: x ** (1 / 3), -2, +2, -2, +2, 200, 200)
    cplot.save_fig("root4.png", lambda x: x ** 0.25, -2, +2, -2, +2, 200, 200)

    cplot.save_fig("log.png", numpy.log, -2, +2, -2, +2, 200, 200)
    cplot.save_fig("exp.png", numpy.exp, -2, +2, -2, +2, 200, 200)
    cplot.save_fig("exp1z.png", lambda z: numpy.exp(1 / z), -1, +1, -1, +1, 200, 200)

    cplot.save_fig("sin.png", numpy.sin, -5, +5, -5, +5, 200, 200)
    cplot.save_fig("cos.png", numpy.cos, -5, +5, -5, +5, 200, 200)
    cplot.save_fig("tan.png", numpy.tan, -5, +5, -5, +5, 200, 200)

    cplot.save_fig("sinh.png", numpy.sinh, -5, +5, -5, +5, 200, 200)
    cplot.save_fig("cosh.png", numpy.cosh, -5, +5, -5, +5, 200, 200)
    cplot.save_fig("tanh.png", numpy.tanh, -5, +5, -5, +5, 200, 200)

    cplot.save_fig("arcsin.png", numpy.arcsin, -2, +2, -2, +2, 200, 200)
    cplot.save_fig("arccos.png", numpy.arccos, -2, +2, -2, +2, 200, 200)
    cplot.save_fig("arctan.png", numpy.arctan, -2, +2, -2, +2, 200, 200)

    cplot.save_fig("gamma.png", scipy.special.gamma, -5, +5, -5, +5, 200, 200)
    cplot.save_fig("digamma.png", scipy.special.digamma, -5, +5, -5, +5, 200, 200)
    cplot.save_fig("zeta.png", zeta, -30, +30, -30, +30, 200, 200)

    # First function from the SIAM-100-digit challenge
    # <https://en.wikipedia.org/wiki/Hundred-dollar,_Hundred-digit_Challenge_problems>
    def siam(z):
        return numpy.cos(numpy.log(z) / z) / z

    cplot.save_fig("siam.png", siam, -1, 1, -1, 1, 230, 230, alpha=0.1)

    # a = 10
    # cplot.save_fig("bessel0.png", lambda z: scipy.special.jv(0, z), -a, +a, -a, +a, 100, 100)
    # cplot.save_fig("bessel1.png", lambda z: scipy.special.jv(1, z), -a, +a, -a, +a, 100, 100)
    # cplot.save_fig("bessel2.png", lambda z: scipy.special.jv(2, z), -a, +a, -a, +a, 100, 100)
    # cplot.save_fig("bessel3.png", lambda z: scipy.special.jv(3, z), -a, +a, -a, +a, 100, 100)
    return


def test_compare_colorspaces():
    def f(z):
        return (z ** 2 - 1) * (z - 2 - 1j) ** 2 / (z ** 2 + 2 + 2j)

    n = 201
    cplot.save_fig("cam16-10.png", f, -3, +3, -3, +3, n, n, colorspace="cam16")
    cplot.save_fig("cielab-10.png", f, -3, +3, -3, +3, n, n, colorspace="cielab")
    cplot.save_fig("hsl-10.png", f, -3, +3, -3, +3, n, n, colorspace="hsl")

    cplot.save_fig("cam16-05.png", f, -3, +3, -3, +3, n, n, 0.5, "cam16")
    cplot.save_fig("cielab-05.png", f, -3, +3, -3, +3, n, n, 0.5, "cielab")
    cplot.save_fig("hsl-05.png", f, -3, +3, -3, +3, n, n, 0.5, "hsl")

    cplot.save_fig("cam16-00.png", f, -3, +3, -3, +3, n, n, 0, "cam16")
    cplot.save_fig("cielab-00.png", f, -3, +3, -3, +3, n, n, 0, "cielab")
    cplot.save_fig("hsl-00.png", f, -3, +3, -3, +3, n, n, 0, "hsl")
    return


if __name__ == "__main__":
    # test_cam16()
    test_compare_colorspaces()
