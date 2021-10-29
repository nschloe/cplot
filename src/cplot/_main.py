from __future__ import annotations

from typing import Callable

import matplotlib as mpl
import matplotlib.pyplot as plt
import mplx
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


def _plot_colors(
    fz,
    extent,
    abs_scaling: Callable[[np.ndarray], np.ndarray] = lambda r: r / (r + 1),
    colorspace: str = "cam16",
    add_colorbars: bool | tuple[bool, bool] = True,
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
        # Don't use "nearest" interpolation, it creates color blocking artifacts:
        # <https://github.com/matplotlib/matplotlib/issues/21499>
        # interpolation="nearest",
        origin="lower",
        aspect="equal",
    )

    if isinstance(add_colorbars, bool):
        add_colorbars = (add_colorbars, add_colorbars)

    if add_colorbars[1]:
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
            pad=0.08 if add_colorbars[0] else 0.04,
        )
        cb1.set_label("arg", rotation=0, ha="center", va="top")
        cb1.ax.yaxis.set_label_coords(0.5, -0.03)
        cb1.set_ticks([-np.pi, -np.pi / 2, 0, +np.pi / 2, np.pi])
        cb1.set_ticklabels(["-π", "-π/2", "0", "π/2", "π"])

    if add_colorbars[0]:
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


def _plot_contour_abs(
    Z,
    fz,
    contours: ArrayLike | float = 2.0,
    emphasize_contour_1: bool = True,
    alpha: float = 1.0,
    color: str | None = None,
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

    if isinstance(contours, (float, int)):
        base = contours
        min_exp = np.log(np.min(vals)) / np.log(base)
        min_exp = int(max(min_exp, -100))
        max_exp = np.log(np.max(vals)) / np.log(base)
        max_exp = int(min(max_exp, 100))
        contours_neg = [base ** k for k in range(min_exp, 0)]
        contours_pos = [base ** k for k in range(1, max_exp + 1)]

        _plot_contour(contours_neg, color if color else "0.8", "solid", alpha)
        if emphasize_contour_1:
            # subtle emphasize
            _plot_contour([1.0], "0.6", "solid", 0.7)
            # "dash":
            # _plot_contour([1.0], "0.8", [(0, (5, 5))], 0.2)
            # _plot_contour([1.0], "0.3", [(5, (5, 5))], 0.2)
        else:
            _plot_contour([1.0], color if color else "0.8", "solid", alpha)

        _plot_contour(contours_pos, color if color else "0.3", "solid", alpha)
    else:
        contours = np.asarray(contours)
        _plot_contour(contours, color if color else "0.8", "solid", alpha)


def _plot_contour_arg(
    Z,
    fz,
    contours: ArrayLike = (-np.pi / 2, 0.0, np.pi / 2, np.pi),
    colorspace: str = "CAM16",
    saturation_adjustment: float = 1.28,
    max_jump: float = 1.0,
    lightness_adjustment: float = 1.0,
    alpha: float = 1.0,
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

    for contours, angle_fun in [
        (contours1, np.angle),
        (contours2, lambda z: np.mod(np.angle(z), 2 * np.pi)),
    ]:
        contours = np.asarray(contours)

        if len(contours) == 0:
            continue

        linecolors = get_srgb1(
            lightness_adjustment * np.exp(contours * 1j),
            abs_scaling=lambda r: r / (r + 1),
            colorspace=colorspace,
            saturation_adjustment=saturation_adjustment,
        )

        mplx.contour(
            Z.real,
            Z.imag,
            angle_fun(fz),
            levels=contours,
            colors=linecolors,
            alpha=alpha,
            max_jump=max_jump,
        )
    plt.gca().set_aspect("equal")


def plot(
    f: Callable[[np.ndarray], np.ndarray],
    x_range: tuple[float, float, int],
    y_range: tuple[float, float, int],
    # abs_scaling: Callable[[np.ndarray], np.ndarray] = lambda r: r ** 2 / (r ** 2 + 1),
    abs_scaling: Callable[[np.ndarray], np.ndarray] = lambda r: r / (r + 1),
    contours_abs: float | ArrayLike | None = 2.0,
    contours_arg: ArrayLike | None = (-np.pi / 2, 0, np.pi / 2, np.pi),
    contour_arg_max_jump: float = 1.0,
    emphasize_abs_contour_1: bool = True,
    colorspace: str = "cam16",
    add_colorbars: bool | tuple[bool, bool] = True,
    add_axes_labels: bool = True,
    saturation_adjustment: float = 1.28,
):
    Z = _get_z_grid_for_image(x_range, y_range)
    fz = f(Z)

    extent = (x_range[0], x_range[1], y_range[0], y_range[1])
    _plot_colors(
        fz,
        extent,
        abs_scaling,
        colorspace,
        add_colorbars=add_colorbars,
        saturation_adjustment=saturation_adjustment,
    )

    if contours_abs is not None:
        _plot_contour_abs(
            Z,
            fz,
            contours=contours_abs,
            emphasize_contour_1=emphasize_abs_contour_1,
            alpha=0.2,
        )

    if contours_arg is not None:
        _plot_contour_arg(
            Z,
            fz,
            contours=contours_arg,
            colorspace=colorspace,
            saturation_adjustment=saturation_adjustment,
            max_jump=contour_arg_max_jump,
            alpha=0.4,
            # Draw the arg contour lines a little lighter. This way, arg contours which
            # dissolve into areas of nearly equal arg remain recognizable. (E.g., tangent,
            # zeta, erf,...).
            lightness_adjustment=1.5,
        )

    if add_axes_labels:
        plt.xlabel("Re(z)")
        # ylabel off-center, <https://github.com/matplotlib/matplotlib/issues/21467>
        plt.ylabel(
            "Im(z)",
            rotation="horizontal",
            loc="center",
            verticalalignment="center",
            labelpad=10,
        )
    return plt


# only show the absolute value
def plot_abs(
    *args,
    add_colorbars: bool = True,
    contours_abs: str | ArrayLike | None = None,
    **kwargs
):
    return plot(
        *args,
        contours_abs=contours_abs,
        contours_arg=None,
        emphasize_abs_contour_1=False,
        add_colorbars=(add_colorbars, False),
        saturation_adjustment=0.0,
        **kwargs
    )


# only show the phase, with some default value adjustments
def plot_arg(*args, add_colorbars: bool = True, **kwargs):
    return plot(
        *args,
        abs_scaling=lambda r: np.full_like(r, 0.5),
        contours_abs=None,
        contours_arg=None,
        emphasize_abs_contour_1=False,
        add_colorbars=(False, add_colorbars),
        **kwargs
    )


# "Phase plot" is a common name for this kind of plots
plot_phase = plot_abs


# only show the phase, with some default value adjustments
def plot_contours(
    f: Callable[[np.ndarray], np.ndarray],
    x_range: tuple[float, float, int],
    y_range: tuple[float, float, int],
    contours_abs: float | ArrayLike | None = 2.0,
    contours_arg: ArrayLike | None = (-np.pi / 2, 0, np.pi / 2, np.pi),
    colorspace: str = "cam16",
    contour_arg_max_jump: float = 1.0,
    saturation_adjustment: float = 1.28,
):
    Z = _get_z_grid_for_image(x_range, y_range)
    fz = f(Z)

    if contours_arg is not None:
        _plot_contour_arg(
            Z,
            fz,
            contours=contours_arg,
            colorspace=colorspace,
            saturation_adjustment=saturation_adjustment,
            max_jump=contour_arg_max_jump,
            alpha=1.0,
            lightness_adjustment=1.5,
        )

    if contours_abs is not None:
        _plot_contour_abs(
            Z,
            fz,
            contours=contours_abs,
            alpha=0.8,
            color="0.7",
            emphasize_contour_1=False,
        )

    plt.gca().set_aspect("equal")
