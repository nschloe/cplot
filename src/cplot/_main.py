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
        # contours: Optional[Union[ntp.ArrayLike, Literal["auto"]]] = "auto",
        contours: Optional[Union[ntp.ArrayLike, str]] = "auto",
        linecolors: str = "#a0a0a0",
        linestyles: str = "solid",
        linestyles_abs1: str = "dashed",
    ):
        if contours is None:
            return

        vals = np.abs(self.fz)

        if contours == "auto":
            base = 2.0
            min_exp = np.log(np.min(vals)) / np.log(base)
            min_exp = int(max(min_exp, -100))
            max_exp = np.log(np.max(vals)) / np.log(base)
            max_exp = int(min(max_exp, 100))
            contours = [base ** k for k in range(min_exp, max_exp + 1) if k != 0]

        contours = np.asarray(contours)

        plt.contour(
            self.Z.real,
            self.Z.imag,
            vals,
            levels=contours,
            colors=linecolors,
            linestyles=linestyles,
            alpha=0.4,
        )
        # give the option to let abs==1 have a different line style
        plt.contour(
            self.Z.real,
            self.Z.imag,
            np.abs(self.fz),
            levels=[1],
            colors=linecolors,
            linestyles=linestyles_abs1,
            alpha=0.4,
        )
        plt.gca().set_aspect("equal")

    def plot_contour_arg(
        self,
        contours: Optional[ntp.ArrayLike] = (-np.pi / 2, 0, np.pi / 2, np.pi),
        linecolors="#a0a0a050",
        linestyles="solid",
        colorspace: str = "CAM16",
    ):
        if contours is None:
            return

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
            (contours2, _angle2, (0.0, 2 * np.pi)),
        ]:
            if len(contours) == 0:
                continue

            linecolors = get_srgb1(
                np.exp(contours * 1j), abs_scaling="h-1.0", colorspace=colorspace
            )

            c = plt.contour(
                self.Z.real,
                self.Z.imag,
                angle_fun(self.fz),
                levels=contours,
                colors=linecolors,
                linestyles=linestyles,
                alpha=0.4,
            )
            for level, allseg in zip(contours, c.allsegs):
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
    contours=("auto", (-np.pi / 2, 0, np.pi / 2, np.pi)),
    linecolors="#a0a0a050",
    linestyles="solid",
):
    plotter = Plotter(f, xminmax, yminmax, n)

    plotter.plot_contour_abs(
        levels=contours[0],
        linecolors=linecolors,
        linestyles=linestyles,
    )
    plotter.plot_contour_arg(
        levels=contours[1],
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
    contours=("auto", (-np.pi / 2, 0, np.pi / 2, np.pi)),
    colorspace: str = "cam16",
    linecolors: str = "#a0a0a050",
    linestyles: str = "solid",
    linestyles_abs1: str = "solid",
    colorbars: bool = True,
):
    plotter = Plotter(f, xminmax, yminmax, n)
    plotter.plot_colors(abs_scaling, colorspace, add_colorbars=colorbars)

    plotter.plot_contour_abs(
        levels=contours[0],
        linecolors=linecolors,
        linestyles=linestyles,
        linestyles_abs1=linestyles_abs1,
    )
    plotter.plot_contour_arg(
        levels=contours[1],
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
