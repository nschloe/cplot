from __future__ import annotations

from typing import Callable

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import ArrayLike

from ._colors import get_srgb1


def _get_z_grid_for_image(
    xspec: tuple[float, float, int], yspec: tuple[float, float, int]
) -> np.ndarray:
    xmin, xmax, nx = xspec
    ymin, ymax, ny = yspec
    assert xmin < xmax
    assert ymin < ymax

    hx = (xmax - xmin) / nx
    x = np.linspace(xmin + hx / 2, xmax - hx / 2, nx)
    hy = (ymax - ymin) / ny
    y = np.linspace(ymin + hy / 2, ymax - hy / 2, ny)

    X = np.meshgrid(x, y)
    return X[0] + 1j * X[1]


def plot_colors(
    fz,
    extent,
    abs_scaling: Callable[[np.ndarray], np.ndarray] = lambda x: x / (x + 1),
    colorspace: str = "cam16",
    add_colorbars: bool = True,
    saturation_adjustment: float = 1.28,
):
    plt.imshow(
        get_srgb1(
            fz,
            abs_scaling=abs_scaling,
            colorspace=colorspace,
            saturation_adjustment=saturation_adjustment,
        ),
        extent=extent,
        interpolation="nearest",
        origin="lower",
        aspect="equal",
    )

    if add_colorbars:
        # arg colorbar
        # create new colormap
        z = np.exp(1j * np.linspace(-np.pi, np.pi, 256))
        rgb_vals = get_srgb1(
            z,
            abs_scaling=abs_scaling,
            colorspace=colorspace,
            saturation_adjustment=saturation_adjustment,
        )
        rgba_vals = np.pad(rgb_vals, ((0, 0), (0, 1)), constant_values=1.0)
        newcmp = mpl.colors.ListedColormap(rgba_vals)
        #
        norm = mpl.colors.Normalize(vmin=-np.pi, vmax=np.pi)
        cb1 = plt.colorbar(
            mpl.cm.ScalarMappable(norm=norm, cmap=newcmp),
            fraction=0.046,
            pad=0.08,
        )
        cb1.set_label("arg", rotation=0, ha="center", va="top")
        cb1.ax.yaxis.set_label_coords(0.5, -0.03)
        cb1.set_ticks([-np.pi, -np.pi / 2, 0, +np.pi / 2, np.pi])
        cb1.set_ticklabels(["-π", "-π/2", "0", "π/2", "π"])

        # abs colorbar
        norm = mpl.colors.Normalize(vmin=0, vmax=1)
        cb0 = plt.colorbar(
            mpl.cm.ScalarMappable(norm=norm, cmap=mpl.cm.gray),
            # This works okay-ish trying to match the height of the colorbar with that
            # of the axes. <https://stackoverflow.com/a/26720422/353337>
            fraction=0.046,
            pad=0.04,
        )
        cb0.set_label("abs", rotation=0, ha="center", va="top")
        cb0.ax.yaxis.set_label_coords(0.5, -0.03)
        scaled_vals = abs_scaling(np.array([1 / 8, 1 / 4, 1 / 2, 1, 2, 4, 8]))
        cb0.set_ticks([0.0, *scaled_vals, 1.0])
        cb0.set_ticklabels(["0", "1/8", "1/4", "1/2", "1", "2", "4", "8", "∞"])


def plot_contour_abs(
    Z,
    fz,
    # Literal["auto"] needs Python 3.8
    contours: ArrayLike | str = "auto",
    highlight_contour_1: bool = True,
):
    vals = np.abs(fz)

    def _plot_contour(levels, colors, linestyles, alpha):
        plt.contour(
            Z.real,
            Z.imag,
            vals,
            levels=levels,
            colors=colors,
            linestyles=linestyles,
            alpha=alpha,
        )

    if contours == "auto":
        base = 2.0
        min_exp = np.log(np.min(vals)) / np.log(base)
        min_exp = int(max(min_exp, -100))
        max_exp = np.log(np.max(vals)) / np.log(base)
        max_exp = int(min(max_exp, 100))
        contours_neg = [base ** k for k in range(min_exp, 0)]
        contours_pos = [base ** k for k in range(1, max_exp + 1)]

        _plot_contour(contours_neg, "0.8", "solid", 0.2)
        if highlight_contour_1:
            # subtle highlight
            _plot_contour([1.0], "0.6", "solid", 0.7)
            # "dash":
            # _plot_contour([1.0], "0.8", [(0, (5, 5))], 0.2)
            # _plot_contour([1.0], "0.3", [(5, (5, 5))], 0.2)
        else:
            _plot_contour([1.0], "0.8", "solid", 0.2)

        _plot_contour(contours_pos, "0.3", "solid", 0.2)
    else:
        contours = np.asarray(contours)
        _plot_contour(contours, "0.8", "solid", 0.2)

    plt.gca().set_aspect("equal")


def plot_contour_arg(
    Z,
    fz,
    f: Callable[[np.ndarray], np.ndarray],
    contours: ArrayLike = (-np.pi / 2, 0.0, np.pi / 2, np.pi),
    colorspace: str = "CAM16",
    saturation_adjustment: float = 1.28,
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

    for contours, angle_fun, branch_cut in [
        (contours1, np.angle, (-np.pi, np.pi)),
        (contours2, lambda z: np.mod(np.angle(z), 2 * np.pi), (0.0, 2 * np.pi)),
    ]:
        contours = np.asarray(contours)

        if len(contours) == 0:
            continue

        # Draw the arg contour lines a little lighter. This way, arg contours which
        # dissolve into areas of nearly equal arg remain recognizable. (E.g., tangent,
        # zeta, erf,...)
        lightness_adjustment = 1.5

        linecolors = get_srgb1(
            lightness_adjustment * np.exp(contours * 1j),
            abs_scaling=lambda x: x / (x + 1),
            colorspace=colorspace,
            saturation_adjustment=saturation_adjustment,
        )

        c = plt.contour(
            Z.real,
            Z.imag,
            angle_fun(fz),
            levels=contours,
            colors=linecolors,
            linestyles="solid",
            alpha=0.4,
        )
        # plt.contour draws some lines in excess which need to be cut off. This is
        # done via setting some values to NaN, see
        # <https://github.com/matplotlib/matplotlib/issues/20548>.
        for level, allseg in zip(contours, c.allsegs):
            for segment in allseg:
                x, y = segment.T
                z = x + 1j * y
                angle = angle_fun(f(z))
                # cut off segments close to the branch cut
                is_near_branch_cut = np.logical_or(
                    *[np.abs(angle - bc) < np.abs(angle - level) for bc in branch_cut]
                )
                segment[is_near_branch_cut] = np.nan
    plt.gca().set_aspect("equal")


def plot(
    f: Callable[[np.ndarray], np.ndarray],
    x_range: tuple[float, float, int],
    y_range: tuple[float, float, int],
    # abs_scaling: Callable[[np.ndarray], np.ndarray] = lambda x: x ** 2 / (x ** 2+ 1),
    abs_scaling: Callable[[np.ndarray], np.ndarray] = lambda x: x / (x + 1),
    contours_abs: str | ArrayLike | None = "auto",
    contours_arg: ArrayLike | None = (-np.pi / 2, 0, np.pi / 2, np.pi),
    highlight_abs_contour_1: bool = True,
    colorspace: str = "cam16",
    add_colorbars: bool = True,
    saturation_adjustment: float = 1.28,
):
    Z = _get_z_grid_for_image(x_range, y_range)
    fz = f(Z)
    extent = (x_range[0], x_range[1], y_range[0], y_range[1])
    plot_colors(
        fz,
        extent,
        abs_scaling,
        colorspace,
        add_colorbars=add_colorbars,
        saturation_adjustment=saturation_adjustment,
    )
    if contours_abs is not None:
        plot_contour_abs(
            Z, fz, contours=contours_abs, highlight_contour_1=highlight_abs_contour_1
        )
    if contours_arg is not None:
        plot_contour_arg(
            Z, fz, f, contours=contours_arg, saturation_adjustment=saturation_adjustment
        )
    return plt
