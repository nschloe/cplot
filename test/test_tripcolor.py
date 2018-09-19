# -*- coding: utf-8 -*-
#
import matplotlib.pyplot as plt
import matplotlib.tri as tri
import numpy

import cplot


def test_tripcolor():
    # Adapted from
    # <https://matplotlib.org/gallery/images_contours_and_fields/tripcolor_demo.html#sphx-glr-gallery-images-contours-and-fields-tripcolor-demo-py>
    # First create the x and y coordinates of the points.
    n_angles = 36
    n_radii = 8
    min_radius = 0.25
    radii = numpy.linspace(min_radius, 0.95, n_radii)

    angles = numpy.linspace(0, 2 * numpy.pi, n_angles, endpoint=False)
    angles = numpy.repeat(angles[..., numpy.newaxis], n_radii, axis=1)
    angles[:, 1::2] += numpy.pi / n_angles

    x = (radii * numpy.cos(angles)).flatten()
    y = (radii * numpy.sin(angles)).flatten()
    z = 2 * (x + 1j * y)

    # Create the Triangulation; no triangles so Delaunay triangulation created.
    triang = tri.Triangulation(x, y)

    # Mask off unwanted triangles.
    triang.set_mask(
        numpy.hypot(x[triang.triangles].mean(axis=1), y[triang.triangles].mean(axis=1))
        < min_radius
    )

    cplot.tripcolor(triang, z)
    plt.gca().set_aspect("equal", "datalim")
    plt.show()
    return


if __name__ == "__main__":
    test_tripcolor()
