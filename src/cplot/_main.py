from __future__ import annotations

import warnings
from typing import Callable

import matplotlib.pyplot as plt
import matplotx
import numpy as np
from matplotlib import cm, colors
from mpl_toolkits.axes_grid1 import make_axes_locatable
from numpy.typing import ArrayLike

from ._colors import get_srgb1, lightness_r, scale_abs, srgb1_grey


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
    abs_scaling: Callable[[np.ndarray], np.ndarray] | None = None,
    saturation_adjustment: float = 1.28,
):
    rgb_vals = get_srgb1(
        fz,
        abs_scaling=abs_scaling,
        saturation_adjustment=saturation_adjustment,
    )

    # set nan values to white
    assert rgb_vals.shape[-1] == 3
    is_nan = np.any(np.isnan(rgb_vals), axis=-1)
    rgb_vals[is_nan] = [1.0, 1.0, 1.0]

    plt.imshow(
        rgb_vals,
        extent=extent,
        # Don't use "nearest" interpolation, it creates color blocking artifacts:
        # <https://github.com/matplotlib/matplotlib/issues/21499>
        # interpolation="nearest",
        origin="lower",
        aspect="equal",
    )


def _add_colorbar_arg(cax, saturation_adjustment: float):
    # arg colorbar
    # create new colormap
    z = np.exp(1j * np.linspace(-np.pi, np.pi, 256))
    rgb_vals = get_srgb1(
        z,
        abs_scaling=lambda z: np.full_like(z, 0.5),
        saturation_adjustment=saturation_adjustment,
    )
    rgba_vals = np.pad(rgb_vals, ((0, 0), (0, 1)), constant_values=1.0)
    newcmp = colors.ListedColormap(rgba_vals)
    #
    norm = colors.Normalize(vmin=-np.pi, vmax=np.pi)

    cb1 = plt.colorbar(cm.ScalarMappable(norm=norm, cmap=newcmp), cax=cax)

    cb1.set_label("arg", rotation=0, ha="center", va="top")
    cb1.ax.yaxis.set_label_coords(0.5, -0.03)
    cb1.set_ticks([-np.pi, -np.pi / 2, 0, +np.pi / 2, np.pi])
    cb1.set_ticklabels(
        [r"$-\pi$", r"$-\dfrac{\pi}{2}$", "$0$", r"$\dfrac{\pi}{2}$", r"$\pi$"]
    )


def _add_colorbar_abs(cax, abs_scaling: Callable, abs_contours: float | list[float]):
    # abs colorbar; use the actual (grey) lightness values of the plot
    s = np.linspace(0.0, 1.0, 256)
    greys = get_srgb1(np.ones(256), abs_scaling=lambda _: s, saturation_adjustment=0.0)
    greys = np.pad(greys, ((0, 0), (0, 1)), constant_values=1.0)
    norm = colors.Normalize(vmin=0, vmax=1)
    cb0 = plt.colorbar(
        cm.ScalarMappable(norm=norm, cmap=colors.ListedColormap(greys)),
        cax=cax,
    )
    cb0.set_label("abs", rotation=0, ha="center", va="top")
    cb0.ax.yaxis.set_label_coords(0.5, -0.03)

    if isinstance(abs_contours, (int, float)):
        a = f"{abs_contours:g}"
        scaled_vals = abs_scaling(
            np.array([abs_contours**k for k in range(-3, 4)], dtype=float)
        )
        cb0.set_ticks([0.0, *scaled_vals, 1.0])
        if isinstance(abs_contours, int) and abs_contours < 4:
            cb0.set_ticklabels(
                [
                    "0",
                    f"$\\frac{{1}}{{{abs_contours**3}}}$",
                    f"$\\frac{{1}}{{{abs_contours**2}}}$",
                    f"$\\frac{{1}}{{{abs_contours**1}}}$",
                    "$1$",
                    f"{abs_contours**1}",
                    f"{abs_contours**2}",
                    f"{abs_contours**3}",
                    "$\\infty$",
                ]
            )
        else:
            cb0.set_ticklabels(
                [
                    "$0$",
                    f"${a}^{{-3}}$",
                    f"${a}^{{-2}}$",
                    f"${a}^{{-1}}$",
                    "$1$",
                    f"${a}^1$",
                    f"${a}^2$",
                    f"${a}^3$",
                    "$\\infty$",
                ]
            )
    else:
        abs_contours = np.asarray(abs_contours, dtype=float)
        scaled_vals = abs_scaling(abs_contours)
        cb0.set_ticks([0.0, *scaled_vals, 1.0])
        cb0.set_ticklabels(["0", *[f"{val:g}" for val in abs_contours], "∞"])


def _contour_abs_colors(
    levels: ArrayLike,
    abs_scaling: Callable[[np.ndarray], np.ndarray],
    alpha: float,
    sign: ArrayLike,
    contrast: float = 0.09,
) -> list[str]:
    """Grey line colors for the contours |f| = level.

    The background lightness along |f| = c is the same for all arguments, namely that
    of abs_scaling(c). Pick a grey that, drawn with opacity alpha, is `contrast`
    lighter (sign > 0) or darker (sign < 0) than the background, measured in
    toe-corrected OKLab lightness L_r. This makes all contours equally visible,
    including the ones close to |f| = 1 where the background is mid-grey.
    """
    s = abs_scaling(np.asarray(levels, dtype=float))
    lr = lightness_r(s)
    background = srgb1_grey(lr)
    target = srgb1_grey(np.clip(lr + np.sign(sign) * contrast, 0.0, 1.0))
    # matplotlib blends in sRGB: out = (1 - alpha) * background + alpha * line
    line = np.clip((target - (1.0 - alpha) * background) / alpha, 0.0, 1.0)
    return [colors.to_hex((g, g, g)) for g in line]


def _plot_contour_abs(
    Z,
    fz,
    contours: ArrayLike | float = 2.0,
    emphasize_contour_1: bool = True,
    alpha: float = 1.0,
    # in each direction, positive and negative:
    max_num_contours: int = 100,
    color: str | None = None,
    min_contour_length: float | None = None,
    linewidth: float | None = None,
    # If given, the line colors are adapted to the background lightness
    abs_scaling: Callable[[np.ndarray], np.ndarray] | None = None,
):
    vals = np.abs(fz)

    def _plot_contour(levels, colors, linestyles, alpha):
        if len(levels) == 0:
            return
        matplotx.contour(
            Z.real,
            Z.imag,
            vals,
            levels=levels,
            colors=colors,
            linestyles=linestyles,
            alpha=alpha,
            min_contour_length=min_contour_length,
            # choose a minjump above machine precision; avoids
            # speckles for functions like `z / abs(z)`
            min_jump=1.0e-15,
            linewidths=linewidth,
        )

    def _colors(levels, default):
        if color:
            return color
        if abs_scaling is None:
            return default
        # lighter lines where |f| <= 1 (dark background), darker where |f| > 1
        levels = np.asarray(levels, dtype=float)
        sign = np.where(levels <= 1.0, 1.0, -1.0)
        return _contour_abs_colors(levels, abs_scaling, alpha, sign)

    if isinstance(contours, (float, int)):
        base = contours

        minval = np.nanmin(vals)
        min_exp = -np.inf if minval == 0.0 else np.log(minval) / np.log(base)

        mx = max(min_exp, -max_num_contours)
        min_exp = 0 if np.isnan(mx) else int(mx)

        maxval = np.nanmax(vals)
        max_exp = np.log(maxval) / np.log(base)
        mn = min(max_exp, max_num_contours)
        max_exp = 0 if np.isnan(mn) else int(mn)

        # exclude exponent 0, that's treated separately below
        contours_neg = [base**k for k in range(min_exp, 0)]
        contours_pos = [base**k for k in range(1, max_exp + 1)]

        _plot_contour(contours_neg, _colors(contours_neg, "0.8"), "solid", alpha)

        if color is None and abs_scaling is not None:
            # |f| = 1 is drawn like the |f| < 1 contours (light); it is recognizable
            # as the line where the contours switch from light to dark.
            light = _contour_abs_colors([1.0], abs_scaling, alpha, +1.0)
            _plot_contour([1.0], light, "solid", alpha)
        elif emphasize_contour_1 and color is None:
            # subtle emphasize
            _plot_contour([1.0], "0.6", "solid", 0.7)
        else:
            _plot_contour([1.0], color if color else "0.8", "solid", alpha)

        _plot_contour(contours_pos, _colors(contours_pos, "0.3"), "solid", alpha)
    else:
        contours = np.asarray(contours)
        _plot_contour(contours, _colors(contours, "0.8"), "solid", alpha)


def _plot_contour_arg(
    Z,
    fz,
    angles: ArrayLike = (-np.pi / 2, 0.0, np.pi / 2, np.pi),
    saturation_adjustment: float = 1.28,
    max_jump: float = 1.0,
    lightness_adjustment: float = 1.0,
    alpha: float = 1.0,
    min_contour_length: float | None = None,
    linewidth: float | None = None,
):
    angles = np.asarray(angles)

    # assert angles in [-pi, pi], like np.angle
    angles = np.mod(angles + np.pi, 2 * np.pi) - np.pi

    # Contour contours must be increasing
    angles = np.sort(angles)

    # mpl has problems with plotting the contour at +pi because that's where the
    # branch cut in np.angle happens. Separate out this case and move the branch cut
    # to 0/2*pi there.
    is_level1 = (angles > -np.pi + 0.1) & (angles < np.pi - 0.1)
    angles1 = angles[is_level1]
    angles2 = angles[~is_level1]
    angles2 = np.mod(angles2, 2 * np.pi)

    for angles, angle_fun in [
        (angles1, np.angle),
        (angles2, lambda z: np.mod(np.angle(z), 2 * np.pi)),
    ]:
        angles = np.asarray(angles)

        if len(angles) == 0:
            continue

        linecolors = get_srgb1(
            lightness_adjustment * np.exp(angles * 1j),
            abs_scaling=lambda r: r / (r + 1),
            saturation_adjustment=saturation_adjustment,
        )

        matplotx.contour(
            Z.real,
            Z.imag,
            angle_fun(fz),
            levels=list(angles),
            colors=list(linecolors),
            min_contour_length=min_contour_length,
            alpha=alpha,
            max_jump=max_jump,
            linewidths=linewidth,
        )
    plt.gca().set_aspect("equal")


class Plotter:
    def __init__(
        self, x_range: tuple[float, float, int], y_range: tuple[float, float, int]
    ):
        self.Z = _get_z_grid_for_image(x_range, y_range)
        self.extent = (x_range[0], x_range[1], y_range[0], y_range[1])

    def plot(self, fz, *args, **kwargs):
        return _plot(self.Z, fz, self.extent, *args, **kwargs)


def _evaluate(f: Callable[[np.ndarray], np.ndarray], Z: np.ndarray) -> np.ndarray:
    # always pass a flat vector, makes it easier for f(); broadcast so that functions
    # returning a scalar (constants) work, too
    fz = np.asarray(f(Z.ravel()))
    return np.broadcast_to(fz, (Z.size,)).reshape(Z.shape)


def plot(
    f: Callable[[np.ndarray], np.ndarray],
    x_range: tuple[float, float, int],
    y_range: tuple[float, float, int],
    *args,
    **kwargs,
):
    extent = (x_range[0], x_range[1], y_range[0], y_range[1])
    Z = _get_z_grid_for_image(x_range, y_range)

    fz = _evaluate(f, Z)

    _plot(Z, fz, extent, *args, **kwargs)
    return plt


def _flat_lightness(r: np.ndarray) -> np.ndarray:
    # constant lightness, used for plot_arg()
    return np.full_like(r, 0.5, dtype=float)


def _warn_abs_scaling(abs_scaling, stacklevel: int = 3) -> None:
    if abs_scaling is not None and abs_scaling is not _flat_lightness:
        warnings.warn(
            "abs_scaling is deprecated and will be removed in a future version. "
            "cplot uses a fixed lightness scaling so that plots are comparable.",
            DeprecationWarning,
            stacklevel=stacklevel,
        )


def _resolve_abs_scaling(abs_scaling) -> Callable:
    if abs_scaling is None:
        return scale_abs
    return (
        abs_scaling if callable(abs_scaling) else _abs_scaling_from_float(abs_scaling)
    )


def _abs_scaling_from_float(val: float) -> Callable:
    assert val > 1
    alpha = np.log(2) / np.log(val)

    def alpha_scaling(r):
        return r**alpha / (r**alpha + 1)

    return alpha_scaling


def _plot(
    Z: np.ndarray,
    fz: np.ndarray,
    extent: tuple[float, float, float, float],
    # deprecated; plots use a fixed lightness scaling (cplot.scale_abs)
    abs_scaling: float | Callable[[np.ndarray], np.ndarray] | None = None,
    # Literal["auto"]
    contours_abs: float | list[float] | None | str = "auto",
    contours_arg: ArrayLike | None = (-np.pi / 2, 0, np.pi / 2, np.pi),
    contour_arg_max_jump: float = 1.0,
    emphasize_abs_contour_1: bool = True,
    add_colorbars: bool | tuple[bool, bool] = True,
    colorbar_pad: tuple[float, float] = (0.2, 0.5),
    add_axes_labels: bool = True,
    saturation_adjustment: float = 1.28,
    min_contour_length: float | None = None,
    linewidth: float | None = None,
):
    assert Z.shape == fz.shape

    # user -> plot() / Plotter.plot() -> _plot() -> _warn_abs_scaling()
    _warn_abs_scaling(abs_scaling, stacklevel=4)
    asc = _resolve_abs_scaling(abs_scaling)

    _plot_colors(
        fz,
        extent,
        asc,
        saturation_adjustment=saturation_adjustment,
    )

    if isinstance(contours_abs, str) and contours_abs == "auto":
        # contours at powers of 2, or at powers of the (deprecated) float abs_scaling
        contours_abs = abs_scaling if isinstance(abs_scaling, (int, float)) else 2

    if contours_abs is not None:
        _plot_contour_abs(
            Z,
            fz,
            contours=contours_abs,
            emphasize_contour_1=emphasize_abs_contour_1,
            alpha=0.4,
            abs_scaling=asc,
            min_contour_length=min_contour_length,
            linewidth=linewidth,
        )

    if contours_arg is not None:
        _plot_contour_arg(
            Z,
            fz,
            angles=contours_arg,
            saturation_adjustment=saturation_adjustment,
            max_jump=contour_arg_max_jump,
            alpha=0.4,
            # Draw the arg contour lines a little lighter. This way, arg contours which
            # dissolve into areas of nearly equal arg remain recognizable. (E.g., tan,
            # zeta, erf,...).
            lightness_adjustment=1.5,
            min_contour_length=min_contour_length,
            linewidth=linewidth,
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

    # colorbars?
    if isinstance(add_colorbars, bool):
        add_colorbars = (add_colorbars, add_colorbars)

    ax = plt.gca()
    divider = make_axes_locatable(ax)

    if add_colorbars[0]:
        cax1 = divider.append_axes("right", size="5%", pad=colorbar_pad[0])
        # without abs contours, still label the colorbar at powers of 2
        _add_colorbar_abs(cax1, asc, 2 if contours_abs is None else contours_abs)

    if add_colorbars[1]:
        cax2 = divider.append_axes("right", size="5%", pad=colorbar_pad[1])
        _add_colorbar_arg(cax2, saturation_adjustment)
    return plt


# only show the absolute value
def plot_abs(
    *args,
    add_colorbars: bool = True,
    contours_abs: str | float | list[float] | None = "auto",
    **kwargs,
):
    return plot(
        *args,
        contours_abs=contours_abs,
        contours_arg=None,
        emphasize_abs_contour_1=False,
        add_colorbars=(add_colorbars, False),
        saturation_adjustment=0.0,
        **kwargs,
    )


# only show the phase, with some default value adjustments
def plot_arg(*args, add_colorbars: bool = True, **kwargs):
    return plot(
        *args,
        abs_scaling=_flat_lightness,
        contours_abs=None,
        contours_arg=None,
        emphasize_abs_contour_1=False,
        add_colorbars=(False, add_colorbars),
        **kwargs,
    )


# "Phase plot" is a common name for this kind of plots
plot_phase = plot_arg


# only show the contours, no colors
def plot_contours(
    f: Callable[[np.ndarray], np.ndarray],
    x_range: tuple[float, float, int],
    y_range: tuple[float, float, int],
    contours_abs: float | ArrayLike | None = 2,
    contours_arg: ArrayLike | None = (-np.pi / 2, 0, np.pi / 2, np.pi),
    contour_arg_max_jump: float = 1.0,
    saturation_adjustment: float = 1.28,
):
    Z = _get_z_grid_for_image(x_range, y_range)

    fz = _evaluate(f, Z)

    if contours_arg is not None:
        _plot_contour_arg(
            Z,
            fz,
            angles=contours_arg,
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
    return plt
