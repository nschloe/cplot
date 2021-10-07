from __future__ import annotations

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


def tricontour_abs(triang, fz: ArrayLike, contours: ArrayLike | None = None):
    vals = np.abs(fz)

    def plot_contours(levels, colors, linestyles, alpha):
        plt.tricontour(
            triang,
            vals,
            levels=levels,
            colors=colors,
            linestyles=linestyles,
            alpha=alpha,
        )

    if contours is None:
        base = 2.0
        min_exp = np.log(np.min(vals)) / np.log(base)
        min_exp = int(max(min_exp, -100))
        max_exp = np.log(np.max(vals)) / np.log(base)
        max_exp = int(min(max_exp, 100))
        contours_neg = [base ** k for k in range(min_exp, 0)]
        contours_pos = [base ** k for k in range(1, max_exp + 1)]

        plot_contours(levels=contours_neg, colors="0.8", linestyles="solid", alpha=0.2)
        plot_contours([1.0], colors="0.8", linestyles=[(5, (5, 5))], alpha=0.3)
        plot_contours([1.0], colors="0.3", linestyles=[(0, (5, 5))], alpha=0.3)
        plot_contours(levels=contours_pos, colors="0.3", linestyles="solid", alpha=0.2)
    else:
        plot_contours(levels=contours, colors="0.8", linestyles="solid", alpha=0.2)

    return plt


def tricontour_arg(
    triang,
    fz: ArrayLike,
    # f: Callable[[np.ndarray], np.ndarray],
    contours: ArrayLike = (-np.pi / 2, 0.0, np.pi / 2, np.pi),
    colorspace: str = "CAM16",
):
    contours = np.asarray(contours)

    # assert contours in [-pi, pi], like np.angle
    contours = np.mod(contours + np.pi, 2 * np.pi) - np.pi

    # Contour contours must be increasing
    contours = np.sort(contours)

    # mpl has problems with plotting the contour at +pi because that's where the
    # branch cut in np.angle happens. Separate out this case and move the branch cut
    # to 0/2*pi there.
    is_level1 = (contours > -np.pi + 0.1) & (contours < np.pi - 0.1)
    contours1 = contours[is_level1]
    contours2 = contours[~is_level1]
    contours2 = np.mod(contours2, 2 * np.pi)

    # plt.contour draws some lines in excess which need to be cut off. This is done
    # via setting some values to NaN, see
    # <https://github.com/matplotlib/matplotlib/issues/20548>.
    for contours, angle_fun, branch_cut in [
        (contours1, np.angle, (-np.pi, np.pi)),
        (contours2, lambda z: np.mod(np.angle(z), 2 * np.pi), (0.0, 2 * np.pi)),
    ]:
        if len(contours) == 0:
            continue

        linecolors = get_srgb1(
            np.exp(contours * 1j),
            abs_scaling=lambda x: x / (x + 1),
            colorspace=colorspace,
        )

        plt.tricontour(
            triang,
            angle_fun(fz),
            levels=contours,
            colors=linecolors,
            linestyles="solid",
            alpha=0.4,
        )
        # for level, allseg in zip(contours, c.allsegs):
        #     for segment in allseg:
        #         x, y = segment.T
        #         z = x + 1j * y
        #         angle = angle_fun(f(z))
        #         # cut off segments close to the branch cut
        #         is_near_branch_cut = np.logical_or(
        #             *[np.abs(angle - bc) < np.abs(angle - level) for bc in branch_cut]
        #         )
        #         segment[is_near_branch_cut] = np.nan
    plt.gca().set_aspect("equal")
