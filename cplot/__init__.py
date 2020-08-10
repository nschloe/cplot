from .__about__ import __version__
from .benchmark import show_test_function
from .create import create_colormap, find_max_srgb_radius, show_circular, show_linear

__all__ = [
    "__version__",
    #
    "show_test_function",
    "show_linear",
    "show_circular",
    "find_max_srgb_radius",
    "create_colormap",
    "show",
    "plot",
    "save_fig",
    "save_img",
    "get_srgb1",
    #
    "tripcolor",
]
