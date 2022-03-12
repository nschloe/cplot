import numpy as np

import cplot


def test_array():
    np.random.seed(0)
    n = 5
    z = np.random.rand(n) + 1j * np.random.rand(n)
    vals = cplot.get_srgb1(z)
    assert vals.shape == (n, 3)
