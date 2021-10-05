import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

from ._colors import get_srgb1


def tripcolor(triang, z, abs_scaling: str = "h-1"):
    rgb = get_srgb1(z, abs_scaling=abs_scaling)

    # https://github.com/matplotlib/matplotlib/issues/10265#issuecomment-358684592
    n = z.shape[0]
    z2 = np.arange(n)
    cmap = mpl.colors.LinearSegmentedColormap.from_list("mymap", rgb, N=n)
    plt.tripcolor(triang, z2, shading="gouraud", cmap=cmap)
