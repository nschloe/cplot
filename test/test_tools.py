# -*- coding: utf-8 -*-
#
import cplot


def test_create():
    vals = cplot.create_colormap()
    cplot.show_linear(vals)
    return


if __name__ == '__main__':
    test_create()
