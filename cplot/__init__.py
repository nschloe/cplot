# -*- coding: utf-8 -*-
#
from .__about__ import (
    __author__,
    __author_email__,
    __copyright__,
    __license__,
    __version__,
    __maintainer__,
    __status__,
)


from .create import (
    show_kovesi_test_image,
    show_linear,
    show_circular,
    find_max_srgb_radius,
    create_colormap,
)
from .main import show, plot, savefig
from .tripcolor import tripcolor

__all__ = [
    "__author__",
    "__author_email__",
    "__copyright__",
    "__license__",
    "__version__",
    "__maintainer__",
    "__status__",
    #
    "show_kovesi_test_image",
    "show_linear",
    "show_circular",
    "find_max_srgb_radius",
    "create_colormap",
    "show",
    "plot",
    "savefig",
    #
    "tripcolor",
]
