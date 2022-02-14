import numpy as np

import cplot


def test_riemann():
    cplot.riemann_sphere(np.log, off_screen=True)
