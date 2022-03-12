import matplotlib.pyplot as plt
import numpy as np


def show_linear(vals) -> None:
    plt.imshow(np.multiply.outer(np.ones(60), vals.data.T))
    plt.show()


def show_circular(vals, rot=0.0):
    n = 256
    x, y = np.meshgrid(np.linspace(-n, +n), np.linspace(-n, +n))

    alpha = np.mod(np.arctan2(y, x) - rot, 2 * np.pi)

    m = vals.data.shape[1]
    ls = np.linspace(0, 2 * np.pi, m, endpoint=False)
    r = np.interp(alpha.reshape(-1), ls, vals.data[0]).reshape(alpha.shape)
    g = np.interp(alpha.reshape(-1), ls, vals.data[1]).reshape(alpha.shape)
    b = np.interp(alpha.reshape(-1), ls, vals.data[2]).reshape(alpha.shape)
    out = np.array([r, g, b])

    plt.imshow(out.T)
    plt.show()


def find_max_srgb_radius(cs, L=50, tol=1.0e-6):
    from colorio.cs import ColorCoordinates, convert

    # In the given color space find the circle in the L=50-plane with the center (50, 0,
    # 0) such that it's as large as possible while still being in the SRGB gamut.
    n = 256
    alpha = np.linspace(0, 2 * np.pi, n, endpoint=False)

    # bisection
    r0 = 0.0
    r1 = 100.0
    while r1 - r0 > tol:
        r = 0.5 * (r1 + r0)

        coords = ColorCoordinates(
            [np.full(n, L), r * np.cos(alpha), r * np.sin(alpha)], cs
        )
        vals = convert(coords, "srgb1", mode="ignore")

        if np.any(vals < 0) or np.any(vals > 1):
            r1 = r
        else:
            r0 = r
    return r0


def create_colormap(L=50):
    import colorio
    from colorio.cs import SRGB1, ColorCoordinates, convert

    cs = colorio.cs.CAM16UCS(c=0.69, Y_b=20, L_A=15)
    # cs = colorio.cs.CAM02('UCS', 0.69, 20, 15)
    # cs = colorio.cs.CIELAB()

    r0 = find_max_srgb_radius(cs, L=L)

    n = 256
    alpha = np.linspace(0, 2 * np.pi, n, endpoint=False)

    coords = ColorCoordinates(
        [np.full(n, L), r0 * np.cos(alpha), r0 * np.sin(alpha)], cs
    )
    return convert(coords, SRGB1("clip"))
