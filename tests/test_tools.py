import matplotlib
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


def test_array():
    numpy.random.seed(0)
    n = 5
    z = numpy.random.rand(n) + 1j * numpy.random.rand(n)
    vals = cplot.get_srgb1(z)
    assert vals.shape == (n, 3)


if __name__ == "__main__":
    test_create()
