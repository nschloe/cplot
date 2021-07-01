from typing import Callable, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as ntp

from ._colors import get_abs_scaling_arctan, get_abs_scaling_h, get_srgb1


class Plot:
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

    def __del__(self):
        plt.close()

    def plot_colors(
        self,
        abs_scaling: str = "h-1.0",
        colorspace: str = "cam16",
    ):
        plt.imshow(
            get_srgb1(self.fz, abs_scaling=abs_scaling, colorspace=colorspace),
            extent=self.extent,
            interpolation="nearest",
            origin="lower",
            aspect="equal",
        )

    def plot_contour_abs(
        self,
        levels: Union[int, ntp.ArrayLike] = 7,
        colors="#a0a0a050",
        linestyles="solid",
    ):
        if levels in [None, 0]:
            return

        if isinstance(levels, int):
            levels = [2.0 ** k for k in np.arange(0, levels) - levels // 2]

        levels = np.asarray(levels)

        plt.contour(
            self.Z.real,
            self.Z.imag,
            np.abs(self.fz),
            levels=levels,
            colors=colors,
            linestyles=linestyles,
        )
        plt.gca().set_aspect("equal")

    def plot_contour_arg(
        self,
        levels: Union[int, ntp.ArrayLike] = 4,
        colors="#a0a0a050",
        linestyles="solid",
    ):
        if levels in [None, 0]:
            return

        if isinstance(levels, int):
            levels = np.linspace(0.0, 2 * np.pi, levels, endpoint=False)
        else:
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

        # plt.contour draws some lines in excess, which need to be cut off. This is done
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
                colors=colors,
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

    def show(self):
        plt.show()

    def savefig(self, filename):
        plt.savefig(filename, transparent=True, bbox_inches="tight")


def show_colors(
    f: Callable,
    xminmax: Tuple[float, float],
    yminmax: Tuple[float, float],
    n: Union[int, Tuple[int, int]],
    abs_scaling="h-1.0",
    colorspace="cam16",
):
    plot = Plot(f, xminmax, yminmax, n)
    plot.plot_colors(abs_scaling, colorspace)
    plot.show()


def show_contours(
    f: Callable,
    xminmax: Tuple[float, float],
    yminmax: Tuple[float, float],
    n: Union[int, Tuple[int, int]],
    levels=(7, 4),
    colors="#a0a0a050",
    linestyles="solid",
):
    plot = Plot(f, xminmax, yminmax, n)

    plot.plot_contour_abs(
        levels=levels[0],
        colors=colors,
        linestyles=linestyles,
    )
    plot.plot_contour_arg(
        levels=levels[1],
        colors=colors,
        linestyles=linestyles,
    )
    plot.show()


def plot(
    f: Callable,
    xminmax: Tuple[float, float],
    yminmax: Tuple[float, float],
    n: Union[int, Tuple[int, int]],
    abs_scaling: str = "h-1.0",
    colorspace: str = "cam16",
    levels=(7, 4),
    colors="#a0a0a050",
    linestyles="solid",
):
    plot = Plot(f, xminmax, yminmax, n)

    plot.plot_colors(abs_scaling, colorspace)

    if levels in [None, 0]:
        levels = (0, 0)

    plot.plot_contour_abs(
        levels=levels[0],
        colors=colors,
        linestyles=linestyles,
    )
    plot.plot_contour_arg(
        levels=levels[1],
        colors=colors,
        linestyles=linestyles,
    )
    return plot


def show(*args, **kwargs):
    plot(*args, **kwargs).show()


def savefig(filename, *args, **kwargs):
    plot(*args, **kwargs).savefig(filename)


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
