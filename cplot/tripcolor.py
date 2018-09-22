# -*- coding: utf-8 -*-
#
import matplotlib
import matplotlib.pyplot as plt
import numpy

from .main import get_srgb


def tripcolor(triang, z, abs_scaling=lambda r: r / (r + 1)):
    angle = numpy.arctan2(z.imag, z.real)
    absval_scaled = abs_scaling(numpy.abs(z))
    rgb = get_srgb(angle, absval_scaled)

    # https://github.com/matplotlib/matplotlib/issues/10265#issuecomment-358684592
    n = z.shape[0]
    z2 = numpy.arange(n)
    cmap = matplotlib.colors.LinearSegmentedColormap.from_list("mymap", rgb, N=n)
    plt.tripcolor(triang, z2, shading="gouraud", cmap=cmap)
    return
