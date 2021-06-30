from .__about__ import __version__
from ._colors import get_srgb1, tripcolor
from ._main import Plot, imsave, plot_contours, savefig, show
from .benchmark import show_kovesi_test_image, show_test_function
from .create import create_colormap, find_max_srgb_radius, show_circular, show_linear

__all__ = [
    "__version__",
    #
    "show_test_function",
    "show_kovesi_test_image",
    "show_linear",
    "show_circular",
    "find_max_srgb_radius",
    "create_colormap",
    "show",
    "plot",
    "savefig",
    "imsave",
    "get_srgb1",
    "plot_contours",
    "Plot",
    #
    "tripcolor",
]
