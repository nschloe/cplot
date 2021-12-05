from typing import Callable

import matplotlib.pyplot as plt
import numpy as np
import scipyx as spx
from mpmath import fp
from scipy.special import (
    airy,
    airye,
    digamma,
    erf,
    exp1,
    expi,
    fresnel,
    gamma,
    jn,
    lambertw,
    sici,
    wofz,
)

import cplot

# gray to improve visibility on github's dark background
_gray = "#969696"
style = {
    "text.color": _gray,
    "axes.labelcolor": _gray,
    "axes.edgecolor": _gray,
    "xtick.color": _gray,
    "ytick.color": _gray,
}
plt.style.use(style)


def _wrap(fun: Callable) -> Callable:
    def wrapped_fun(z):
        z = np.asarray(z)
        z_shape = z.shape
        out = np.array([fun(complex(val)) for val in z.flatten()])
        return out.reshape(z_shape)

    return wrapped_fun


def riemann_xi(z):
    # https://en.wikipedia.org/wiki/Riemann_Xi_function
    return 0.5 * z * (z - 1) * np.pi ** (-z / 2) * gamma(z / 2) * _wrap(fp.zeta)(z)


def f(z):
    return (z ** 2 - 1) * (z - 2 - 1j) ** 2 / (z ** 2 + 2 + 2j)


# n = 201
# for name in ["cam16", "cielab", "oklab", "hsl"]:
#     cplot.plot(f, (-3, +3, n), (-3, +3, n), colorspace=name, add_colorbars=False)
#     plt.savefig(f"{name}-10.svg", transparent=True, bbox_inches="tight")
#     plt.close()
#     #
#     cplot.plot(
#         f,
#         (-3, +3, n),
#         (-3, +3, n),
#         colorspace=name,
#         abs_scaling=2.0,
#         add_colorbars=False,
#     )
#     plt.savefig(f"{name}-05.svg", transparent=True, bbox_inches="tight")
#     plt.close()
#     #
#     cplot.plot(
#         f,
#         (-3, +3, n),
#         (-3, +3, n),
#         colorspace=name,
#         abs_scaling=lambda x: np.full_like(x, 0.5),
#         add_colorbars=False,
#     )
#     plt.savefig(f"{name}-00.svg", transparent=True, bbox_inches="tight")
#     plt.close()


# First function from the SIAM-100-digit challenge
# <https://en.wikipedia.org/wiki/Hundred-dollar,_Hundred-digit_Challenge_problems>
n = 401
cplot.plot(
    lambda z: np.cos(np.log(z) / z) / z, (-1, 1, n), (-1, 1, n), abs_scaling=10.0
)
plt.savefig("siam.svg", transparent=True, bbox_inches="tight")
plt.close()

n = 400
cplot.plot_abs(lambda z: np.sin(z ** 3) / z, (-2, 2, n), (-2, 2, n))
plt.savefig("sinz3z-abs.svg", bbox_inches="tight")
plt.close()

cplot.plot_arg(lambda z: np.sin(z ** 3) / z, (-2, 2, n), (-2, 2, n))
plt.savefig("sinz3z-arg.svg", bbox_inches="tight")
plt.close()

cplot.plot_contours(lambda z: np.sin(z ** 3) / z, (-2, 2, n), (-2, 2, n))
plt.savefig("sinz3z-contours.svg", bbox_inches="tight")
plt.close()

cplot.plot(lambda z: np.sin(z ** 3) / z, (-2, 2, n), (-2, 2, n))
plt.savefig("sinz3z.svg", transparent=True, bbox_inches="tight")
plt.close()


args = [
    #
    ("z1.svg", lambda z: z ** 1, (-2, +2), (-2, +2)),
    ("z2.svg", lambda z: z ** 2, (-2, +2), (-2, +2)),
    ("z3.svg", lambda z: z ** 3, (-2, +2), (-2, +2)),
    #
    ("1z.svg", lambda z: 1 / z, (-2.0, +2.0), (-2.0, +2.0)),
    ("1z2.svg", lambda z: 1 / z ** 2, (-2.0, +2.0), (-2.0, +2.0)),
    ("1z3.svg", lambda z: 1 / z ** 3, (-2.0, +2.0), (-2.0, +2.0)),
    # möbius
    ("moebius1.svg", lambda z: (z + 1) / (z - 1), (-5, +5), (-5, +5)),
    (
        "moebius2.svg",
        lambda z: (z + 1.5 - 0.5j) * (1.5 - 0.5j) / (z - 1.5 + 0.5j) * (-1.5 + 0.5j),
        (-5, +5),
        (-5, +5),
    ),
    (
        "moebius3.svg",
        lambda z: (-1.0j * z) / (1.0j * z + 1.5 - 0.5j),
        (-5, +5),
        (-5, +5),
    ),
    #
    # roots of unity
    ("z6+1.svg", lambda z: z ** 6 + 1, (-1.5, 1.5), (-1.5, 1.5)),
    ("z6-1.svg", lambda z: z ** 6 - 1, (-1.5, 1.5), (-1.5, 1.5)),
    ("z-6+1.svg", lambda z: z ** (-6) + 1, (-1.5, 1.5), (-1.5, 1.5)),
    #
    ("zz.svg", lambda z: z ** z, (-3, +3), (-3, +3)),
    ("1zz.svg", lambda z: (1 / z) ** z, (-3, +3), (-3, +3)),
    ("z1z.svg", lambda z: z ** (1 / z), (-3, +3), (-3, +3)),
    #
    ("root2.svg", np.sqrt, (-2, +2), (-2, +2)),
    ("root3.svg", lambda x: x ** (1 / 3), (-2, +2), (-2, +2)),
    ("root4.svg", lambda x: x ** 0.25, (-2, +2), (-2, +2)),
    #
    ("log.svg", np.log, (-2, +2), (-2, +2)),
    ("exp.svg", np.exp, (-3, +3), (-3, +3)),
    ("exp2.svg", np.exp2, (-3, +3), (-3, +3)),
    #
    # non-analytic functions
    ("re.svg", np.real, (-2, +2), (-2, +2)),
    # ("abs.svg", np.abs, (-2, +2), (-2, +2)),
    ("z-absz.svg", lambda z: z / np.abs(z), (-2, +2), (-2, +2)),
    ("conj.svg", np.conj, (-2, +2), (-2, +2)),
    #
    # essential singularities
    ("exp1z.svg", lambda z: np.exp(1 / z), (-1, +1), (-1, +1)),
    ("zsin1z.svg", lambda z: z * np.sin(1 / z), (-0.6, +0.6), (-0.6, +0.6)),
    ("cos1z.svg", lambda z: np.cos(1 / z), (-0.6, +0.6), (-0.6, +0.6)),
    #
    ("exp-z2.svg", lambda z: np.exp(-(z ** 2)), (-3, +3), (-3, +3)),
    ("11z2.svg", lambda z: 1 / (1 + z ** 2), (-3, +3), (-3, +3)),
    ("erf.svg", erf, (-3, +3), (-3, +3)),
    #
    ("fresnel-s.svg", lambda z: fresnel(z)[0], (-4, +4), (-4, +4)),
    ("fresnel-c.svg", lambda z: fresnel(z)[1], (-4, +4), (-4, +4)),
    ("faddeeva.svg", wofz, (-4, +4), (-4, +4)),
    #
    ("sin.svg", np.sin, (-5, +5), (-5, +5)),
    ("cos.svg", np.cos, (-5, +5), (-5, +5)),
    ("tan.svg", np.tan, (-5, +5), (-5, +5)),
    #
    ("sec.svg", lambda z: 1 / np.cos(z), (-5, +5), (-5, +5)),
    ("csc.svg", lambda z: 1 / np.sin(z), (-5, +5), (-5, +5)),
    ("cot.svg", lambda z: 1 / np.tan(z), (-5, +5), (-5, +5)),
    #
    ("sinh.svg", np.sinh, (-5, +5), (-5, +5)),
    ("cosh.svg", np.cosh, (-5, +5), (-5, +5)),
    ("tanh.svg", np.tanh, (-5, +5), (-5, +5)),
    #
    ("arcsin.svg", np.arcsin, (-2, +2), (-2, +2)),
    ("arccos.svg", np.arccos, (-2, +2), (-2, +2)),
    ("arctan.svg", np.arctan, (-2, +2), (-2, +2)),
    #
    ("sinz-z.svg", lambda z: np.sin(z) / z, (-7, +7), (-7, +7)),
    ("cosz-z.svg", lambda z: np.cos(z) / z, (-7, +7), (-7, +7)),
    ("tanz-z.svg", lambda z: np.tan(z) / z, (-7, +7), (-7, +7)),
    #
    ("si.svg", lambda z: sici(z)[0], (-15, +15), (-15, +15)),
    ("ci.svg", lambda z: sici(z)[1], (-15, +15), (-15, +15)),
    ("expi.svg", expi, (-15, +15), (-15, +15)),
    #
    ("exp1.svg", exp1, (-5, +5), (-5, +5)),
    ("lambertw.svg", lambertw, (-5, +5), (-5, +5)),
    #
    ("gamma.svg", gamma, (-5, +5), (-5, +5)),
    ("digamma.svg", digamma, (-5, +5), (-5, +5)),
    ("zeta.svg", _wrap(fp.zeta), (-30, +30), (-30, +30)),
    #
    ("riemann-xi.svg", riemann_xi, (-20, +20), (-20, +20)),
    ("riemann-siegel-z.svg", _wrap(fp.siegelz), (-20, +20), (-20, +20)),
    ("riemann-siegel-theta.svg", _wrap(fp.siegeltheta), (-20, +20), (-20, +20)),
    #
    # jacobi elliptic functions
    ("ellipj-sn-06.svg", lambda z: spx.ellipj(z, 0.6)[0], (-6, +6), (-6, +6)),
    ("ellipj-cn-06.svg", lambda z: spx.ellipj(z, 0.6)[1], (-6, +6), (-6, +6)),
    ("ellipj-dn-06.svg", lambda z: spx.ellipj(z, 0.6)[2], (-6, +6), (-6, +6)),
    # jacobi theta
    (
        "jtheta1.svg",
        _wrap(lambda z: fp.jtheta(1, z, complex(0.1 * np.exp(0.1j * np.pi)))),
        (-8, +8),
        (-8, +8),
    ),
    (
        "jtheta2.svg",
        _wrap(lambda z: fp.jtheta(2, z, complex(0.1 * np.exp(0.1j * np.pi)))),
        (-8, +8),
        (-8, +8),
    ),
    (
        "jtheta3.svg",
        _wrap(lambda z: fp.jtheta(3, z, complex(0.1 * np.exp(0.1j * np.pi)))),
        (-8, +8),
        (-8, +8),
    ),
    #
    # bessel
    ("bessel-1.svg", lambda z: jn(1, z), (-9, +9), (-9, +9)),
    ("bessel-2.svg", lambda z: jn(2, z), (-9, +9), (-9, +9)),
    ("bessel-3.svg", lambda z: jn(3, z), (-9, +9), (-9, +9)),
    #
    # airy functions
    ("airy-ai.svg", lambda z: airy(z)[0], (-6, +6), (-6, +6)),
    ("airy-bi.svg", lambda z: airy(z)[2], (-6, +6), (-6, +6)),
    ("airye-ai.svg", lambda z: airye(z)[0], (-6, +6), (-6, +6)),
    #
    (
        "tanh-sinh.svg",
        lambda z: np.tanh(np.pi / 2 * np.sinh(z)),
        (-2.5, +2.5),
        (-2.5, +2.5),
    ),
    (
        "sinh-sinh.svg",
        lambda z: np.sinh(np.pi / 2 * np.sinh(z)),
        (-2.5, +2.5),
        (-2.5, +2.5),
    ),
    (
        "exp-sinh.svg",
        lambda z: np.exp(np.pi / 2 * np.sinh(z)),
        (-2.5, +2.5),
        (-2.5, +2.5),
    ),
    #
    # modular forms
    ("kleinj.svg", _wrap(fp.kleinj), (-1.5, +1.5), (1.0e-5, +2.0)),
    ("dedekind-eta.svg", _wrap(fp.eta), (-0.4, +0.4), (1.0e-5, +0.5)),
    #
    # # https://www.dynamicmath.xyz
    # (
    #     "some-polynomial.svg",
    #     lambda z: 0.926 * (z + 7.3857e-2 * z ** 5 + 4.5458e-3 * z ** 9),
    #     (-3, 3),
    #     (-3, 3),
    # ),
    # # non-analytic
    # (
    #     "non-analytic.svg",
    #     lambda z: np.imag(np.exp(-1j * np.pi / 4) * z ** n)
    #     + 1j * np.imag(np.exp(1j * np.pi / 4) * (z - 1) ** 4),
    #     (-2.0, +3.0),
    #     (-2.0, +3.0),
    # ),
    #
]
for filename, fun, x, y in args:
    diag_length = np.sqrt((x[1] - x[0]) ** 2 + (y[1] - y[0]) ** 2)
    cplot.plot(
        fun,
        (x[0], x[1], n),
        (y[0], y[1], n),
        add_colorbars=False,
        add_axes_labels=False,
        min_contour_length=1.0e-2 * diag_length,
    )
    plt.savefig(filename, transparent=True, bbox_inches="tight")
    plt.close()
