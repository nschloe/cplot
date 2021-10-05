import numpy as np

import cplot


def test_basic():
    def f(z):
        return np.sin(z ** 3) / z

    plt = cplot.plot(
        f,
        (-2.0, +2.0),
        (-2.0, +2.0),
        400,
        # colorbars: bool = True,
        # abs_scaling="h-1.0",        # how to scale the lightness in domain coloring
        # colorspace: str = "cam16",  # ditto
        # abs/args contour lines:
        # contours=("auto", (-np.pi / 2, 0, np.pi / 2, np.pi)),
        # linecolors = "#a0a0a050",
        # linestyles = "solid",
        # linestyle_abs1 = "solid"
    )
    plt.show()
