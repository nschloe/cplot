import colorio
import matplotlib.pyplot as plt
import numpy as np


def show_linear(vals):
    plt.imshow(np.multiply.outer(np.ones(60), vals.T))
    plt.show()


def show_circular(vals, rot=0.0):
    n = 256
    x, y = np.meshgrid(np.linspace(-n, +n), np.linspace(-n, +n))

    alpha = np.mod(np.arctan2(y, x) - rot, 2 * np.pi)

    m = vals.shape[1]
    ls = np.linspace(0, 2 * np.pi, m, endpoint=False)
    r = np.interp(alpha.reshape(-1), ls, vals[0]).reshape(alpha.shape)
    g = np.interp(alpha.reshape(-1), ls, vals[1]).reshape(alpha.shape)
    b = np.interp(alpha.reshape(-1), ls, vals[2]).reshape(alpha.shape)
    out = np.array([r, g, b])

    plt.imshow(out.T)
    plt.show()


def find_max_srgb_radius(cs, srgb, L=50, tol=1.0e-6):
    # In the given color space find the circle in the L=50-plane with the center (50, 0,
    # 0) such that it's as large as possible while still being in the SRGB gamut.
    n = 256
    alpha = np.linspace(0, 2 * np.pi, n, endpoint=False)

    # bisection
    r0 = 0.0
    r1 = 100.0
    while r1 - r0 > tol:
        r = 0.5 * (r1 + r0)

        pts = np.array([np.full(n, L), r * np.cos(alpha), r * np.sin(alpha)])
        vals = srgb.from_xyz100(cs.to_xyz100(pts))

        if np.any(vals < 0) or np.any(vals > 1):
            r1 = r
        else:
            r0 = r
    return r0


def create_colormap(L=50):
    cam = colorio.cs.CAM16UCS(0.69, 20, 64 / np.pi / 5)
    # cam = colorio.cs.CAM02('UCS', 0.69, 20, L_A)
    # cam = colorio.cs.CIELAB()
    srgb = colorio.cs.SrgbLinear()

    r0 = find_max_srgb_radius(cam, srgb, L=L)

    n = 256
    alpha = np.linspace(0, 2 * np.pi, n, endpoint=False)

    pts = np.array([np.full(n, L), r0 * np.cos(alpha), r0 * np.sin(alpha)])
    vals = srgb.from_xyz100(cam.to_xyz100(pts))

    # show the colors
    return srgb.to_rgb1(vals)
