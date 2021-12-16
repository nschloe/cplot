import matplotlib
import numpy as np

import cplot


def test_create():
    rgb = cplot.create_colormap(L=50)
    cmap = matplotlib.colors.LinearSegmentedColormap.from_list(
        "custom", rgb.data.T, N=len(rgb.data.T)
    )
    # cmap = 'gray'

    cplot.show_linear(rgb)
    cplot.show_circular(rgb, rot=-np.pi * 4 / 12)
    # cplot.show_circular(rgb, rot=-np.pi * 18/12)
    cplot.show_kovesi_test_image(cmap)
    # cplot.show_kovesi_test_image_circular(cmap)


def test_array():
    np.random.seed(0)
    n = 5
    z = np.random.rand(n) + 1j * np.random.rand(n)
    vals = cplot.get_srgb1(z)
    assert vals.shape == (n, 3)


if __name__ == "__main__":
    test_create()
