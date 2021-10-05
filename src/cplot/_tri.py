from typing import Callable

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import ArrayLike

from ._colors import get_srgb1


def tripcolor(
    triang,
    fz: ArrayLike,
    abs_scaling: Callable[[np.ndarray], np.ndarray] = lambda x: x / (x + 1),
):
    fz = np.asarray(fz)
    rgb = get_srgb1(fz, abs_scaling=abs_scaling)

    # https://github.com/matplotlib/matplotlib/issues/10265#issuecomment-358684592
    n = fz.shape[0]
    z2 = np.arange(n)
    cmap = mpl.colors.LinearSegmentedColormap.from_list("mymap", rgb, N=n)
    plt.tripcolor(triang, z2, shading="gouraud", cmap=cmap)
    return plt


def tri_abs_contour(triang, fz: ArrayLike):
    vals = np.abs(fz)

    base = 2.0
    min_exp = np.log(np.min(vals)) / np.log(base)
    min_exp = int(max(min_exp, -100))
    max_exp = np.log(np.max(vals)) / np.log(base)
    max_exp = int(min(max_exp, 100))
    contours_neg = [base ** k for k in range(min_exp, 0)]
    contours_pos = [base ** k for k in range(1, max_exp + 1)]

    def contours(levels, colors, linestyles, alpha):
        plt.tricontour(
            triang,
            vals,
            levels=levels,
            colors=colors,
            linestyles=linestyles,
            alpha=alpha,
        )

    contours(levels=contours_neg, colors="0.8", linestyles="solid", alpha=0.2)
    contours([1.0], colors="0.3", linestyles=[(5, (5, 5))], alpha=0.3)
    contours([1.0], colors="0.3", linestyles=[(0, (5, 5))], alpha=0.3)
    contours(levels=contours_pos, colors="0.3", linestyles="solid", alpha=0.2)
    return plt
