# -*- coding: utf-8 -*-
#
import numpy

import cplot


def test_create():
    vals = cplot.create_colormap(L=50)
    # cplot.show_linear(vals)
    cplot.show_circular(vals, rot=-numpy.pi * 4/12)
    return


if __name__ == '__main__':
    test_create()
