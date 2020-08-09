import colorio
import matplotlib.pyplot as plt
import numpy


def show_linear(vals):
    plt.imshow(numpy.multiply.outer(numpy.ones(60), vals.T))
    plt.show()


def show_circular(vals, rot=0.0):
    n = 256
    x, y = numpy.meshgrid(numpy.linspace(-n, +n), numpy.linspace(-n, +n))

    alpha = numpy.mod(numpy.arctan2(y, x) - rot, 2 * numpy.pi)

    m = vals.shape[1]
    ls = numpy.linspace(0, 2 * numpy.pi, m, endpoint=False)
    r = numpy.interp(alpha.reshape(-1), ls, vals[0]).reshape(alpha.shape)
    g = numpy.interp(alpha.reshape(-1), ls, vals[1]).reshape(alpha.shape)
    b = numpy.interp(alpha.reshape(-1), ls, vals[2]).reshape(alpha.shape)
    out = numpy.array([r, g, b])

    plt.imshow(out.T)
    plt.show()


def find_max_srgb_radius(cs, srgb, L=50, tol=1.0e-6):
    # In the given color space find the circle in the L=50-plane with the center (50, 0,
    # 0) such that it's as large as possible while still being in the SRGB gamut.
    n = 256
    alpha = numpy.linspace(0, 2 * numpy.pi, n, endpoint=False)

    # bisection
    r0 = 0.0
    r1 = 100.0
    while r1 - r0 > tol:
        r = 0.5 * (r1 + r0)

        pts = numpy.array(
            [numpy.full(n, L), r * numpy.cos(alpha), r * numpy.sin(alpha)]
        )
        vals = srgb.from_xyz100(cs.to_xyz100(pts))

        if numpy.any(vals < 0) or numpy.any(vals > 1):
            r1 = r
        else:
            r0 = r
    return r0


def create_colormap(L=50):
    L_A = 64 / numpy.pi / 5
    cam = colorio.CAM16UCS(0.69, 20, L_A)
    # cam = colorio.CAM02('UCS', 0.69, 20, L_A)
    # cam = colorio.CIELAB()
    srgb = colorio.SrgbLinear()

    r0 = find_max_srgb_radius(cam, srgb, L=L)

    n = 256
    alpha = numpy.linspace(0, 2 * numpy.pi, n, endpoint=False)

    pts = numpy.array([numpy.full(n, L), r0 * numpy.cos(alpha), r0 * numpy.sin(alpha)])
    vals = srgb.from_xyz100(cam.to_xyz100(pts))

    # show the colors
    vals = srgb.to_srgb1(vals)
    return vals
