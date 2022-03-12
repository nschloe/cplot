from .__about__ import __version__
from ._colors import get_srgb1
from ._main import Plotter, plot, plot_abs, plot_arg, plot_contours, plot_phase
from ._riemann_sphere import riemann_sphere
from ._tri import tricontour_abs, tripcolor
from .benchmark import show_kovesi_test_image, show_test_function

__all__ = [
    "show_test_function",
    "show_kovesi_test_image",
    "get_srgb1",
    "plot",
    "plot_arg",
    "plot_abs",
    "plot_phase",
    "plot_contours",
    #
    "riemann_sphere",
    #
    "tripcolor",
    "tricontour_abs",
    #
    "Plotter",
    #
    "__version__",
]
