import matplotlib
import matplotlib.pyplot as plt
import numpy

from .main import get_srgb1


def tripcolor(triang, z):
    rgb = get_srgb1(z)

    # https://github.com/matplotlib/matplotlib/issues/10265#issuecomment-358684592
    n = z.shape[0]
    z2 = numpy.arange(n)
    cmap = matplotlib.colors.LinearSegmentedColormap.from_list("mymap", rgb, N=n)
    plt.tripcolor(triang, z2, shading="gouraud", cmap=cmap)
    return
