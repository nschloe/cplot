from pathlib import Path
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
    hankel1,
    hankel2,
    jn,
    lambertw,
    sici,
    wofz,
    yv,
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

plot_dir = Path(__file__).resolve().parent / ".." / "plots"


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
    return (z**2 - 1) * (z - 2 - 1j) ** 2 / (z**2 + 2 + 2j)


def lambert_1(z, n=100):
    zn = z.copy()
    s = np.zeros_like(z)
    for _ in range(n):
        s += zn / (1 - zn)
        zn *= z

    s[np.abs(z) > 1] = np.nan
    return s


def lambert_phi(z):
    return z / (1 - z) ** 2


def lambert_von_mangoldt(z, n=1000):
    zn = z.copy()
    s = np.zeros_like(z)
    for _ in range(n):
        s += np.log(n) * zn
        zn *= z

    s[np.abs(z) > 1] = np.nan
    return s


def lambert_liouville(z, n=30):
    zk2 = z.copy()
    s = np.zeros_like(z)
    for k in range(n):
        s += zk2
        # zk2 = z ** (k ** 2)
        zk2 *= z ** (2 * k + 1)

    s[np.abs(z) > 1] = np.nan
    return s


# https://en.wikipedia.org/wiki/Euler_function
def euler_function(z, n=1000):
    out = np.ones_like(z)
    zk = z.copy()
    for _ in range(n):
        out *= 1 - zk
        zk *= z

    # Explicitly set some values to nan. This avoids contour artifacts near the
    # boundary.
    out[np.abs(zk) > 1] = np.nan
    return out


# First function from the SIAM-100-digit challenge
# <https://en.wikipedia.org/wiki/Hundred-dollar,_Hundred-digit_Challenge_problems>
n = 401
cplot.plot(
    lambda z: np.cos(np.log(z) / z) / z, (-1, 1, n), (-1, 1, n), abs_scaling=10.0
)
plt.savefig(plot_dir / "siam.png", transparent=True, bbox_inches="tight")
plt.close()

n = 400
cplot.plot_abs(lambda z: np.sin(z**3) / z, (-2, 2, n), (-2, 2, n))
plt.savefig(plot_dir / "sinz3z-abs.png", bbox_inches="tight")
plt.close()

cplot.plot_arg(lambda z: np.sin(z**3) / z, (-2, 2, n), (-2, 2, n))
plt.savefig(plot_dir / "sinz3z-arg.png", bbox_inches="tight")
plt.close()

cplot.plot_contours(lambda z: np.sin(z**3) / z, (-2, 2, n), (-2, 2, n))
plt.savefig(plot_dir / "sinz3z-contours.png", bbox_inches="tight")
plt.close()

cplot.plot(lambda z: np.sin(z**3) / z, (-2, 2, n), (-2, 2, n))
plt.savefig(plot_dir / "sinz3z.png", transparent=True, bbox_inches="tight")
plt.close()


args = [
    #
    ("z1.png", lambda z: z**1, (-2, +2), (-2, +2)),
    ("z2.png", lambda z: z**2, (-2, +2), (-2, +2)),
    ("z3.png", lambda z: z**3, (-2, +2), (-2, +2)),
    #
    ("1z.png", lambda z: 1 / z, (-2.0, +2.0), (-2.0, +2.0)),
    ("1z2.png", lambda z: 1 / z**2, (-2.0, +2.0), (-2.0, +2.0)),
    ("1z3.png", lambda z: 1 / z**3, (-2.0, +2.0), (-2.0, +2.0)),
    # möbius
    ("moebius1.png", lambda z: (z + 1) / (z - 1), (-5, +5), (-5, +5)),
    (
        "moebius2.png",
        lambda z: (z + 1.5 - 0.5j) * (1.5 - 0.5j) / (z - 1.5 + 0.5j) * (-1.5 + 0.5j),
        (-5, +5),
        (-5, +5),
    ),
    (
        "moebius3.png",
        lambda z: (-1.0j * z) / (1.0j * z + 1.5 - 0.5j),
        (-5, +5),
        (-5, +5),
    ),
    #
    # roots of unity
    ("z6+1.png", lambda z: z**6 + 1, (-1.5, 1.5), (-1.5, 1.5)),
    ("z6-1.png", lambda z: z**6 - 1, (-1.5, 1.5), (-1.5, 1.5)),
    ("z-6+1.png", lambda z: z ** (-6) + 1, (-1.5, 1.5), (-1.5, 1.5)),
    #
    ("zz.png", lambda z: z**z, (-3, +3), (-3, +3)),
    ("1zz.png", lambda z: (1 / z) ** z, (-3, +3), (-3, +3)),
    ("z1z.png", lambda z: z ** (1 / z), (-3, +3), (-3, +3)),
    #
    ("root2.png", np.sqrt, (-2, +2), (-2, +2)),
    ("root3.png", lambda x: x ** (1 / 3), (-2, +2), (-2, +2)),
    ("root4.png", lambda x: x**0.25, (-2, +2), (-2, +2)),
    #
    ("log.png", np.log, (-2, +2), (-2, +2)),
    ("exp.png", np.exp, (-3, +3), (-3, +3)),
    ("exp2.png", np.exp2, (-3, +3), (-3, +3)),
    #
    # non-analytic functions
    ("re.png", np.real, (-2, +2), (-2, +2)),
    # ("abs.png", np.abs, (-2, +2), (-2, +2)),
    ("z-absz.png", lambda z: z / np.abs(z), (-2, +2), (-2, +2)),
    ("conj.png", np.conj, (-2, +2), (-2, +2)),
    #
    # essential singularities
    ("exp1z.png", lambda z: np.exp(1 / z), (-1, +1), (-1, +1)),
    ("zsin1z.png", lambda z: z * np.sin(1 / z), (-0.6, +0.6), (-0.6, +0.6)),
    ("cos1z.png", lambda z: np.cos(1 / z), (-0.6, +0.6), (-0.6, +0.6)),
    #
    ("exp-z2.png", lambda z: np.exp(-(z**2)), (-3, +3), (-3, +3)),
    ("11z2.png", lambda z: 1 / (1 + z**2), (-3, +3), (-3, +3)),
    ("erf.png", erf, (-3, +3), (-3, +3)),
    #
    # generating function of fibonacci sequence
    ("fibonacci.png", lambda z: 1 / (1 - z * (1 + z)), (-5.0, +5.0), (-5.0, +5.0)),
    #
    ("fresnel-s.png", lambda z: fresnel(z)[0], (-4, +4), (-4, +4)),
    ("fresnel-c.png", lambda z: fresnel(z)[1], (-4, +4), (-4, +4)),
    ("faddeeva.png", wofz, (-4, +4), (-4, +4)),
    #
    ("sin.png", np.sin, (-5, +5), (-5, +5)),
    ("cos.png", np.cos, (-5, +5), (-5, +5)),
    ("tan.png", np.tan, (-5, +5), (-5, +5)),
    #
    ("sec.png", lambda z: 1 / np.cos(z), (-5, +5), (-5, +5)),
    ("csc.png", lambda z: 1 / np.sin(z), (-5, +5), (-5, +5)),
    ("cot.png", lambda z: 1 / np.tan(z), (-5, +5), (-5, +5)),
    #
    ("sinh.png", np.sinh, (-5, +5), (-5, +5)),
    ("cosh.png", np.cosh, (-5, +5), (-5, +5)),
    ("tanh.png", np.tanh, (-5, +5), (-5, +5)),
    #
    ("arcsin.png", np.arcsin, (-2, +2), (-2, +2)),
    ("arccos.png", np.arccos, (-2, +2), (-2, +2)),
    ("arctan.png", np.arctan, (-2, +2), (-2, +2)),
    #
    ("sinz-z.png", lambda z: np.sin(z) / z, (-7, +7), (-7, +7)),
    ("cosz-z.png", lambda z: np.cos(z) / z, (-7, +7), (-7, +7)),
    ("tanz-z.png", lambda z: np.tan(z) / z, (-7, +7), (-7, +7)),
    #
    ("si.png", lambda z: sici(z)[0], (-15, +15), (-15, +15)),
    ("ci.png", lambda z: sici(z)[1], (-15, +15), (-15, +15)),
    ("expi.png", expi, (-15, +15), (-15, +15)),
    #
    ("exp1.png", exp1, (-5, +5), (-5, +5)),
    ("lambertw.png", lambertw, (-5, +5), (-5, +5)),
    #
    ("gamma.png", gamma, (-5, +5), (-5, +5)),
    ("digamma.png", digamma, (-5, +5), (-5, +5)),
    ("zeta.png", _wrap(fp.zeta), (-30, +30), (-30, +30)),
    #
    ("riemann-xi.png", riemann_xi, (-20, +20), (-20, +20)),
    ("riemann-siegel-z.png", _wrap(fp.siegelz), (-20, +20), (-20, +20)),
    ("riemann-siegel-theta.png", _wrap(fp.siegeltheta), (-20, +20), (-20, +20)),
    #
    # jacobi elliptic functions
    ("ellipj-sn-06.png", lambda z: spx.ellipj(z, 0.6)[0], (-6, +6), (-6, +6)),
    ("ellipj-cn-06.png", lambda z: spx.ellipj(z, 0.6)[1], (-6, +6), (-6, +6)),
    ("ellipj-dn-06.png", lambda z: spx.ellipj(z, 0.6)[2], (-6, +6), (-6, +6)),
    # jacobi theta
    (
        "jtheta1.png",
        _wrap(lambda z: fp.jtheta(1, z, complex(0.1 * np.exp(0.1j * np.pi)))),
        (-8, +8),
        (-8, +8),
    ),
    (
        "jtheta2.png",
        _wrap(lambda z: fp.jtheta(2, z, complex(0.1 * np.exp(0.1j * np.pi)))),
        (-8, +8),
        (-8, +8),
    ),
    (
        "jtheta3.png",
        _wrap(lambda z: fp.jtheta(3, z, complex(0.1 * np.exp(0.1j * np.pi)))),
        (-8, +8),
        (-8, +8),
    ),
    #
    # bessel, first kind
    ("bessel1-1.png", lambda z: jn(1, z), (-9, +9), (-9, +9)),
    ("bessel1-2.png", lambda z: jn(2, z), (-9, +9), (-9, +9)),
    ("bessel1-3.png", lambda z: jn(3, z), (-9, +9), (-9, +9)),
    # bessel, second kind
    ("bessel2-1.png", lambda z: yv(1, z), (-9, +9), (-9, +9)),
    ("bessel2-2.png", lambda z: yv(2, z), (-9, +9), (-9, +9)),
    ("bessel2-3.png", lambda z: yv(3, z), (-9, +9), (-9, +9)),
    #
    # airy functions
    ("airy-ai.png", lambda z: airy(z)[0], (-6, +6), (-6, +6)),
    ("airy-bi.png", lambda z: airy(z)[2], (-6, +6), (-6, +6)),
    ("airye-ai.png", lambda z: airye(z)[0], (-6, +6), (-6, +6)),
    #
    (
        "tanh-sinh.png",
        lambda z: np.tanh(np.pi / 2 * np.sinh(z)),
        (-2.5, +2.5),
        (-2.5, +2.5),
    ),
    (
        "sinh-sinh.png",
        lambda z: np.sinh(np.pi / 2 * np.sinh(z)),
        (-2.5, +2.5),
        (-2.5, +2.5),
    ),
    (
        "exp-sinh.png",
        lambda z: np.exp(np.pi / 2 * np.sinh(z)),
        (-2.5, +2.5),
        (-2.5, +2.5),
    ),
    #
    # modular forms
    ("kleinj.png", _wrap(fp.kleinj), (-2.0, +2.0), (1.0e-5, +2.0)),
    ("dedekind-eta.png", _wrap(fp.eta), (-0.3, +0.3), (1.0e-5, +0.3)),
    # Dedekind eta = Ramanujan Delta ** 24; see
    # https://www.youtube.com/watch?v=s6sdEbGNdic
    # https://en.wikipedia.org/wiki/Ramanujan_tau_function
    #
    # TODO https://en.wikipedia.org/wiki/Euler_function
    # ("euler-function.png", _wrap(fp.eta), (-0.3, +0.3), (1.0e-5, +0.3)),
    #
    ("hankel1a.png", lambda z: hankel1(1.0, z), (-2, +2), (-2, +2)),
    ("hankel1b.png", lambda z: hankel1(3.1, z), (-3, +3), (-3, +3)),
    ("hankel2.png", lambda z: hankel2(1.0, z), (-2, +2), (-2, +2)),
    # lambert series
    ("lambert-1.png", lambert_1, (-1.1, 1.1), (-1.1, 1.1)),
    ("lambert-von-mangoldt.png", lambert_von_mangoldt, (-1.1, 1.1), (-1.1, 1.1)),
    ("lambert-liouville.png", lambert_liouville, (-1.1, 1.1), (-1.1, 1.1)),
    #
    # # https://www.dynamicmath.xyz
    # (
    #     "some-polynomial.png",
    #     lambda z: 0.926 * (z + 7.3857e-2 * z ** 5 + 4.5458e-3 * z ** 9),
    #     (-3, 3),
    #     (-3, 3),
    # ),
    # # non-analytic
    # (
    #     "non-analytic.png",
    #     lambda z: np.imag(np.exp(-1j * np.pi / 4) * z ** n)
    #     + 1j * np.imag(np.exp(1j * np.pi / 4) * (z - 1) ** 4),
    #     (-2.0, +3.0),
    #     (-2.0, +3.0),
    # ),
    # logistic regression:
    ("sigmoid.png", lambda z: 1.0 / (1.0 + np.exp(-z)), (-10, +10), (-10, +10)),
    ("euler-function.png", euler_function, (-1.1, 1.1), (-1.1, 1.1)),
]

for filename, fun, x, y in args:
    diag_length = np.sqrt((x[1] - x[0]) ** 2 + (y[1] - y[0]) ** 2)
    m = int(n * (y[1] - y[0]) / (x[1] - x[0]))
    cplot.plot(
        fun,
        (x[0], x[1], n),
        (y[0], y[1], m),
        add_colorbars=False,
        add_axes_labels=False,
        min_contour_length=1.0e-2 * diag_length,
    )
    plt.savefig(plot_dir / filename, transparent=True, bbox_inches="tight")
    plt.close()
