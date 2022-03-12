from __future__ import annotations

from typing import Callable

import numpy as np

from ._colors import get_srgb1
from ._main import _abs_scaling_from_float


def riemann_sphere(
    f: Callable[[np.ndarray], np.ndarray],
    filename: str | None = None,
    n: int = 50,
    # If you're changing contours_abs to x and want the abs_scaling to follow along,
    # you'll have to set it to the same value.
    abs_scaling: float | Callable[[np.ndarray], np.ndarray] = 2,
    saturation_adjustment: float = 1.28,
    off_screen: bool = False,
) -> None:
    import meshzoo
    import pyvista as pv
    import vtk

    # Use a "flat top" to make sure we never evaluate _exactly_ at infty or 0,
    # just close to it. May save a bit of numerical trouble.
    points, cells = meshzoo.icosa_sphere(n, flat_top=True)

    # stereographic projection onto complex plane
    x, y, z = points.T
    assert np.all(np.abs(x**2 + y**2 + z**2 - 1.0) < 1.0e-13)
    Z = (x + 1j * y) / (1 - z)

    rgb = get_srgb1(
        f(Z),
        abs_scaling if callable(abs_scaling) else _abs_scaling_from_float(abs_scaling),
        saturation_adjustment,
    )

    celltypes = np.full(len(cells), vtk.VTK_TRIANGLE, dtype=np.uint8)
    cells = np.column_stack([np.full(cells.shape[0], cells.shape[1]), cells]).ravel()
    grid = pv.UnstructuredGrid(cells, celltypes, points)
    grid["rgb"] = rgb
    p = pv.Plotter(off_screen=off_screen)
    p.add_mesh(grid, scalars="rgb", rgb=True, lighting=False)
    p.add_axes(xlabel="Re", ylabel="Im", zlabel="abs+")

    return p
