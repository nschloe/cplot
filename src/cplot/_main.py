from typing import Callable, Optional, Tuple, Union

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as ntp

from ._colors import get_srgb1, scale01


class Plotter:
    def __init__(
        self,
        f: Callable,
        xminmax: Tuple[float, float],
        yminmax: Tuple[float, float],
        n: Union[int, Tuple[int, int]],
    ):
        xmin, xmax = xminmax
        ymin, ymax = yminmax
        assert xmin < xmax
        assert ymin < ymax

        if isinstance(n, tuple):
            assert len(n) == 2
            nx, ny = n
        else:
            assert isinstance(n, int)
            nx = n
            ny = n

        self.extent = (xmin, xmax, ymin, ymax)

        self.f = f
        self.Z = _get_z_grid_for_image(xminmax, yminmax, (nx, ny))
        self.fz = f(self.Z)

    # def __del__(self):
    #     plt.close()

    def plot_colors(
        self,
        abs_scaling: str = "h-1.0",
        colorspace: str = "cam16",
        add_colorbars: bool = True,
    ):
        plt.imshow(
            get_srgb1(self.fz, abs_scaling=abs_scaling, colorspace=colorspace),
            extent=self.extent,
            interpolation="nearest",
            origin="lower",
            aspect="equal",
        )

        if add_colorbars:
            # abs colorbar
            norm = mpl.colors.Normalize(vmin=0, vmax=1)
            cb0 = plt.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=mpl.cm.gray))
            cb0.set_label("abs", rotation=0, ha="center", va="top")
            cb0.ax.yaxis.set_label_coords(0.5, -0.03)
            scaled_vals = scale01([1 / 8, 1 / 4, 1 / 2, 1, 2, 4, 8], abs_scaling)
            cb0.set_ticks([0.0, *scaled_vals, 1.0])
            cb0.set_ticklabels(["0", "1/8", "1/4", "1/2", "1", "2", "4", "8", "∞"])

            # arg colorbar
            # create new colormap
            z = np.exp(1j * np.linspace(-np.pi, np.pi, 256))
            rgb_vals = get_srgb1(z, abs_scaling=abs_scaling, colorspace=colorspace)
            rgba_vals = np.pad(rgb_vals, ((0, 0), (0, 1)), constant_values=1.0)
            newcmp = mpl.colors.ListedColormap(rgba_vals)
            #
            norm = mpl.colors.Normalize(vmin=-np.pi, vmax=np.pi)
            cb1 = plt.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=newcmp))
            cb1.set_label("arg", rotation=0, ha="center", va="top")
            cb1.ax.yaxis.set_label_coords(0.5, -0.03)
            cb1.set_ticks([-np.pi, -np.pi / 2, 0, +np.pi / 2, np.pi])
            cb1.set_ticklabels(["-π", "-π/2", "0", "π/2", "π"])

    def plot_contour_abs(
        self,
        # Literal needs Python 3.8
        # levels: Optional[Union[ntp.ArrayLike, Literal["auto"]]] = "auto",
        levels: Optional[Union[ntp.ArrayLike, str]] = "auto",
        linecolors: str = "#a0a0a050",
        linestyles: str = "solid",
        linestyles_abs1: str = "solid",
    ):
        if levels is None:
            return

        vals = np.abs(self.fz)

        if levels == "auto":
            base = 2.0
            k0 = int(np.log(np.min(vals)) / np.log(base))
            k1 = int(np.log(np.max(vals)) / np.log(base))
            levels = [base ** k for k in range(k0, k1) if k != 0]

        levels = np.asarray(levels)

        plt.contour(
            self.Z.real,
            self.Z.imag,
            vals,
            levels=levels,
            colors=linecolors,
            linestyles=linestyles,
        )
        # give the option to let abs==1 have a different line style
        plt.contour(
            self.Z.real,
            self.Z.imag,
            np.abs(self.fz),
            levels=[1],
            colors=linecolors,
            linestyles=linestyles_abs1,
        )
        plt.gca().set_aspect("equal")

    def plot_contour_arg(
        self,
        levels: Optional[ntp.ArrayLike] = (-np.pi / 2, 0, np.pi / 2, np.pi),
        linecolors="#a0a0a050",
        linestyles="solid",
    ):
        if levels is None:
            return

        levels = np.asarray(levels)

        # assert levels in [-pi, pi], like np.angle
        levels = np.mod(levels + np.pi, 2 * np.pi) - np.pi

        # Contour levels must be increasing
        levels = np.sort(levels)

        # mpl has problems with plotting the contour at +pi because that's where the
        # branch cut in np.angle happens. Separate out this case and move the branch cut
        # to 0/2*pi there.
        is_level1 = (levels > -np.pi + 0.1) & (levels < np.pi - 0.1)
        levels1 = levels[is_level1]
        levels2 = levels[~is_level1]
        levels2 = np.mod(levels2, 2 * np.pi)

        # plt.contour draws some lines in excess which need to be cut off. This is done
        # via setting some values to NaN, see
        # <https://github.com/matplotlib/matplotlib/issues/20548>.
        for levels, angle_fun, branch_cut in [
            (levels1, np.angle, (-np.pi, np.pi)),
            (levels2, _angle2, (0.0, 2 * np.pi)),
        ]:
            if len(levels) == 0:
                continue

            c = plt.contour(
                self.Z.real,
                self.Z.imag,
                angle_fun(self.fz),
                levels=levels,
                colors=linecolors,
                linestyles=linestyles,
            )
            for level, allseg in zip(levels, c.allsegs):
                for segment in allseg:
                    x, y = segment.T
                    z = x + 1j * y
                    angle = angle_fun(self.f(z))
                    # cut off segments close to the branch cut
                    is_near_branch_cut = np.logical_or(
                        *[
                            np.abs(angle - bc) < np.abs(angle - level)
                            for bc in branch_cut
                        ]
                    )
                    segment[is_near_branch_cut] = np.nan
        plt.gca().set_aspect("equal")


def plot_colors(
    f: Callable,
    xminmax: Tuple[float, float],
    yminmax: Tuple[float, float],
    n: Union[int, Tuple[int, int]],
    abs_scaling: str = "h-1.0",
    colorspace: str = "cam16",
):
    plotter = Plotter(f, xminmax, yminmax, n)
    plotter.plot_colors(abs_scaling, colorspace)
    return plt


def plot_contours(
    f: Callable,
    xminmax: Tuple[float, float],
    yminmax: Tuple[float, float],
    n: Union[int, Tuple[int, int]],
    levels=("auto", (-np.pi / 2, 0, np.pi / 2, np.pi)),
    linecolors="#a0a0a050",
    linestyles="solid",
):
    plotter = Plotter(f, xminmax, yminmax, n)

    plotter.plot_contour_abs(
        levels=levels[0],
        linecolors=linecolors,
        linestyles=linestyles,
    )
    plotter.plot_contour_arg(
        levels=levels[1],
        linecolors=linecolors,
        linestyles=linestyles,
    )
    return plt


def plot(
    f: Callable,
    xminmax: Tuple[float, float],
    yminmax: Tuple[float, float],
    n: Union[int, Tuple[int, int]] = 500,
    abs_scaling: str = "h-1.0",
    levels=("auto", (-np.pi / 2, 0, np.pi / 2, np.pi)),
    colorspace: str = "cam16",
    linecolors: str = "#a0a0a050",
    linestyles: str = "solid",
    linestyles_abs1: str = "solid",
    colorbars: bool = True,
):
    plotter = Plotter(f, xminmax, yminmax, n)
    plotter.plot_colors(abs_scaling, colorspace, add_colorbars=colorbars)

    plotter.plot_contour_abs(
        levels=levels[0],
        linecolors=linecolors,
        linestyles=linestyles,
        linestyles_abs1=linestyles_abs1,
    )
    plotter.plot_contour_arg(
        levels=levels[1],
        linecolors=linecolors,
        linestyles=linestyles,
    )
    return plt


def _angle2(z):
    return np.mod(np.angle(z), 2 * np.pi)


def _get_z_grid_for_image(
    xminmax: Tuple[float, float],
    yminmax: Tuple[float, float],
    n: Tuple[int, int],
):
    xmin, xmax = xminmax
    ymin, ymax = yminmax
    nx, ny = n

    hx = (xmax - xmin) / nx
    x = np.linspace(xmin + hx / 2, xmax - hx / 2, nx)
    hy = (ymax - ymin) / ny
    y = np.linspace(ymin + hy / 2, ymax - hy / 2, ny)

    X = np.meshgrid(x, y)
    return X[0] + 1j * X[1]
