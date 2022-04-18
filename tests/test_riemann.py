import numpy as np
import pytest

import cplot


def test_riemann():
    pytest.importorskip("pyvista")
    cplot.riemann_sphere(np.log, off_screen=True)
