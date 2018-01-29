# -*- coding: utf-8 -*-
#
import matplotlib
import numpy

import cplot


def test_create():
    rgb = cplot.create_colormap(L=50)
    cmap = matplotlib.colors.LinearSegmentedColormap.from_list(
        'custom', rgb.T, N=len(rgb.T)
        )
    # cmap = 'gray'

    # cplot.show_linear(rgb)
    # cplot.show_circular(rgb, rot=-numpy.pi * 4/12)
    # cplot.show_circular(rgb, rot=-numpy.pi * 18/12)
    cplot.show_kovesi(cmap)
    return


def test_show():
    # cplot.show(lambda z: (z+1)/(z-1), -10, +10, -10, +10, 101, 101)
    # cplot.show(lambda z: (z-1)/(z+1), -4, +4, -4, +4, 200, 200)
    # cplot.show(lambda z: z, -2, +2, -2, +2, 200, 200)
    cplot.show(numpy.tan, -5, +5, -5, +5, 300, 300)
    # cplot.show(numpy.sin, -5, +5, -5, +5, 200, 200)

    # import matplotlib.pyplot as plt
    # cplot.plot(numpy.tan, -5, +5, -5, +5, 300, 300)
    # plt.savefig('out.png', transparent=True)
    return


if __name__ == '__main__':
    test_show()
