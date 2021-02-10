import matplotlib
import mpmath
import numpy

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
    # cplot.show_kovesi_test_image_circular(cmap)


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


def test_compare_colorspaces():
    def f(z):
        return (z ** 2 - 1) * (z - 2 - 1j) ** 2 / (z ** 2 + 2 + 2j)

    names = ["cam16", "cielab", "oklab", "hsl"]

    n = 201
    for name in names:
        cplot.savefig(name + "-10.png", f, -3, +3, -3, +3, n, n, colorspace=name)
        cplot.savefig(name + "-05.png", f, -3, +3, -3, +3, n, n, 0.5, name)
        cplot.savefig(name + "-00.png", f, -3, +3, -3, +3, n, n, 0, name)


if __name__ == "__main__":
    # test_cam16()
    test_compare_colorspaces()
    # test_create()
