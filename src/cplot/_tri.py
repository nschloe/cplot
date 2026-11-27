from __future__ import annotations

import warnings
from typing import Callable

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import ArrayLike

from ._colors import get_srgb1, scale_abs
from ._main import _contour_abs_colors, _resolve_abs_scaling, _warn_abs_scaling


def tripcolor(
    triang,
    fz: ArrayLike,
    # deprecated; plots use a fixed lightness scaling (cplot.scale_abs)
    abs_scaling: Callable[[np.ndarray], np.ndarray] | None = None,
):
    _warn_abs_scaling(abs_scaling)
    fz = np.asarray(fz)
    rgb = get_srgb1(fz, abs_scaling=_resolve_abs_scaling(abs_scaling))

    # https://github.com/matplotlib/matplotlib/issues/10265#issuecomment-358684592
    n = fz.shape[0]
    z2 = np.arange(n)
    cmap = mpl.colors.LinearSegmentedColormap.from_list("mymap", rgb, N=n)
    plt.tripcolor(triang, z2, shading="gouraud", cmap=cmap)
    return plt


def tricontour_abs(triang, fz: ArrayLike, contours: ArrayLike | None = None):
    vals = np.abs(np.asarray(fz))

    # matplotlib's tricontour rejects non-finite values; mask out the triangles that
    # touch them
    is_bad = ~np.isfinite(vals)
    finite_vals = vals[~is_bad]
    if len(finite_vals) == 0:
        return plt
    if np.any(is_bad):
        mask = np.any(is_bad[triang.triangles], axis=1)
        if triang.mask is not None:
            mask |= triang.mask
        triang = mpl.tri.Triangulation(triang.x, triang.y, triang.triangles, mask=mask)
        vals = np.where(is_bad, 1.0, vals)

    alpha = 0.4

    def plot_contours(levels):
        if len(levels) == 0:
            return
        # same line colors as in plot(): constant contrast to the tripcolor background
        sign = np.where(np.asarray(levels) <= 1.0, 1.0, -1.0)
        colors = _contour_abs_colors(levels, scale_abs, alpha, sign)
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore", "No contour levels were found within the data range."
            )
            plt.tricontour(
                triang,
                vals,
                levels=levels,
                colors=colors,
                linestyles="solid",
                alpha=alpha,
            )

    if contours is None:
        base = 2.0
        with np.errstate(divide="ignore"):
            min_exp = int(max(np.log(np.min(finite_vals)) / np.log(base), -100))
            max_exp = int(min(np.log(np.max(finite_vals)) / np.log(base), 100))
        plot_contours([base**k for k in range(min_exp, 0)])
        if np.min(finite_vals) <= 1.0 <= np.max(finite_vals):
            plot_contours([1.0])
        plot_contours([base**k for k in range(1, max_exp + 1)])
    else:
        plot_contours(sorted(np.asarray(contours, dtype=float)))

    return plt


# tricontour_arg is not useful or possible until
# <https://github.com/matplotlib/matplotlib/issues/21309>
#
# def tricontour_arg(
#     triang,
#     fz: ArrayLike,
#     # f: Callable[[np.ndarray], np.ndarray],
#     contours: ArrayLike = (-np.pi / 2, 0.0, np.pi / 2, np.pi),
# ):
