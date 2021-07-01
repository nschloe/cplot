import matplotlib.pyplot as plt
import matplotlib.tri as tri
import numpy as np

import cplot


def create_logo():
    # Adapted from
    # <https://matplotlib.org/gallery/images_contours_and_fields/tripcolor_demo.html#sphx-glr-gallery-images-contours-and-fields-tripcolor-demo-py>
    # First create the x and y coordinates of the points.
    n_angles = 314
    n_radii = 100
    radii = np.linspace(0.0, 1.0, n_radii)

    angles = np.linspace(0, 2 * np.pi, n_angles, endpoint=False)
    angles = np.repeat(angles[..., np.newaxis], n_radii, axis=1)
    angles[:, 1::2] += np.pi / n_angles

    x = (radii * np.cos(angles)).flatten()
    y = (radii * np.sin(angles)).flatten()

    # Create the Triangulation; no triangles so Delaunay triangulation created.
    triang = tri.Triangulation(x, y)

    # print(triang)
    # exit(1)
    # import dmsh
    # geo = dmsh.Circle([0.0, 0.0], 1.0)
    # X, cells = dmsh.generate(geo, 0.1)

    z = x + 1j * y
    # z /= np.abs(z)

    cplot.tripcolor(triang, z, alpha=0)
    plt.gca().set_aspect("equal", "datalim")
    plt.axis("off")

    plt.savefig("logo.png", transparent=True)
    return


if __name__ == "__main__":
    create_logo()
