import matplotlib.pyplot as plt
import matplotlib.tri as tri
import numpy as np

import cplot


def test_tripcolor():
    # Adapted from
    # <https://matplotlib.org/gallery/images_contours_and_fields/tripcolor_demo.html#sphx-glr-gallery-images-contours-and-fields-tripcolor-demo-py>
    # First create the x and y coordinates of the points.
    n_angles = 36
    n_radii = 8
    min_radius = 0.25
    radii = np.linspace(min_radius, 0.95, n_radii)

    angles = np.linspace(0, 2 * np.pi, n_angles, endpoint=False)
    angles = np.repeat(angles[..., np.newaxis], n_radii, axis=1)
    angles[:, 1::2] += np.pi / n_angles

    x = (radii * np.cos(angles)).flatten()
    y = (radii * np.sin(angles)).flatten()
    z = 2 * (x + 1j * y)

    # Create the Triangulation; no triangles so Delaunay triangulation created.
    triang = tri.Triangulation(x, y)

    # Mask off unwanted triangles.
    triang.set_mask(
        np.hypot(x[triang.triangles].mean(axis=1), y[triang.triangles].mean(axis=1))
        < min_radius
    )

    cplot.tripcolor(triang, z)
    plt.gca().set_aspect("equal", "datalim")
    plt.show()


def test_tricontour():
    # Adapted from
    # <https://matplotlib.org/gallery/images_contours_and_fields/tripcolor_demo.html#sphx-glr-gallery-images-contours-and-fields-tripcolor-demo-py>
    # First create the x and y coordinates of the points.
    n_angles = 288
    n_radii = 64
    min_radius = 0.25
    radii = np.linspace(min_radius, 0.95, n_radii)

    angles = np.linspace(0, 2 * np.pi, n_angles, endpoint=False)
    angles = np.repeat(angles[..., np.newaxis], n_radii, axis=1)
    angles[:, 1::2] += np.pi / n_angles

    x = (radii * np.cos(angles)).flatten()
    y = (radii * np.sin(angles)).flatten()
    z = x + 1j * y

    # Create the Triangulation; no triangles so Delaunay triangulation created.
    triang = tri.Triangulation(x, y)

    # Mask off unwanted triangles.
    triang.set_mask(
        np.hypot(x[triang.triangles].mean(axis=1), y[triang.triangles].mean(axis=1))
        < min_radius
    )

    fz = np.sin(7 * z) / (7 * z)

    cplot.tripcolor(triang, fz)
    cplot.tricontour_abs(triang, fz)
    # cplot.tricontour_arg(triang, fz)

    plt.gca().set_aspect("equal", "datalim")
    plt.savefig("out.png", bbox_inches="tight")
    plt.show()


if __name__ == "__main__":
    test_tricontour()
