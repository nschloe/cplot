import numpy as np

import cplot


def test_basic():
    def f(z):
        return np.sin(z**3) / z

    plt = cplot.plot(f, (-2.0, +2.0, 400), (-2.0, +2.0, 400))
    plt.show()
