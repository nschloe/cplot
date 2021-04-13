import matplotlib.pyplot as plt
import mpmath
import numpy as np
import scipy.special

import cplot

# gray to improve visibility on github's dark background
gray = "#969696"
ax = plt.gca()
ax.spines["bottom"].set_color(gray)
ax.spines["top"].set_color(gray)
ax.spines["right"].set_color(gray)
ax.spines["left"].set_color(gray)
ax.tick_params(axis="x", colors=gray)
ax.tick_params(axis="y", colors=gray)


def riemann_zeta(z):
    vals = [[mpmath.zeta(val) for val in row] for row in z]
    out = np.array(
        [[float(val.real) + 1j * float(val.imag) for val in row] for row in vals]
    )
    return out


def riemann_xi(z):
    # https://en.wikipedia.org/wiki/Riemann_Xi_function
    out = (
        0.5
        * z
        * (z - 1)
        * np.pi ** (-z / 2)
        * scipy.special.gamma(z / 2)
        * riemann_zeta(z)
    )
    return out


def riemann_siegel_z(z):
    vals = [[mpmath.siegelz(val) for val in row] for row in z]
    out = np.array(
        [[float(val.real) + 1j * float(val.imag) for val in row] for row in vals]
    )
    return out


def riemann_siegel_theta(z):
    vals = [[mpmath.siegeltheta(val) for val in row] for row in z]
    out = np.array(
        [[float(val.real) + 1j * float(val.imag) for val in row] for row in vals]
    )
    return out


# First function from the SIAM-100-digit challenge
# <https://en.wikipedia.org/wiki/Hundred-dollar,_Hundred-digit_Challenge_problems>
def siam(z):
    return np.cos(np.log(z) / z) / z


n = 400

cplot.savefig("z1.png", lambda z: z ** 1, -2, +2, -2, +2, n, n)
cplot.savefig("z2.png", lambda z: z ** 2, -2, +2, -2, +2, n, n)
cplot.savefig("z3.png", lambda z: z ** 3, -2, +2, -2, +2, n, n)

cplot.savefig("1z.png", lambda z: 1 / z, -2, +2, -2, +2, n, n)
cplot.savefig("z-absz.png", lambda z: z / abs(z), -2, +2, -2, +2, n, n)
cplot.savefig("z+1-z-1.png", lambda z: (z + 1) / (z - 1), -5, +5, -5, +5, n, n)

cplot.savefig("root2.png", np.sqrt, -2, +2, -2, +2, n, n)
cplot.savefig("root3.png", lambda x: x ** (1 / 3), -2, +2, -2, +2, n, n)
cplot.savefig("root4.png", lambda x: x ** 0.25, -2, +2, -2, +2, n, n)

cplot.savefig("log.png", np.log, -2, +2, -2, +2, n, n)
cplot.savefig("exp.png", np.exp, -2, +2, -2, +2, n, n)
cplot.savefig("exp1z.png", lambda z: np.exp(1 / z), -1, +1, -1, +1, n, n)

cplot.savefig("sin.png", np.sin, -5, +5, -5, +5, n, n)
cplot.savefig("cos.png", np.cos, -5, +5, -5, +5, n, n)
cplot.savefig("tan.png", np.tan, -5, +5, -5, +5, n, n)

cplot.savefig("sinh.png", np.sinh, -5, +5, -5, +5, n, n)
cplot.savefig("cosh.png", np.cosh, -5, +5, -5, +5, n, n)
cplot.savefig("tanh.png", np.tanh, -5, +5, -5, +5, n, n)

cplot.savefig("arcsin.png", np.arcsin, -2, +2, -2, +2, n, n)
cplot.savefig("arccos.png", np.arccos, -2, +2, -2, +2, n, n)
cplot.savefig("arctan.png", np.arctan, -2, +2, -2, +2, n, n)

cplot.savefig("sinz-z.png", lambda z: np.sin(z) / z, -7, +7, -7, +7, n, n)
cplot.savefig("cosz-z.png", lambda z: np.cos(z) / z, -7, +7, -7, +7, n, n)
cplot.savefig("tanz-z.png", lambda z: np.tan(z) / z, -7, +7, -7, +7, n, n)

cplot.savefig("gamma.png", scipy.special.gamma, -5, +5, -5, +5, n, n)
cplot.savefig("digamma.png", scipy.special.digamma, -5, +5, -5, +5, n, n)
cplot.savefig("zeta.png", riemann_zeta, -30, +30, -30, +30, n, n)

cplot.savefig("riemann-xi.png", riemann_xi, -20, +20, -20, +20, n, n)
cplot.savefig("riemann-siegel-z.png", riemann_siegel_z, -20, +20, -20, +20, n, n)
cplot.savefig(
    "riemann-siegel-theta.png", riemann_siegel_theta, -20, +20, -20, +20, n, n
)

cplot.savefig("siam.png", siam, -1, 1, -1, 1, n, n, alpha=0.1)


def f(z):
    return (z ** 2 - 1) * (z - 2 - 1j) ** 2 / (z ** 2 + 2 + 2j)


n = 201
for name in ["cam16", "cielab", "oklab", "hsl"]:
    cplot.savefig(f"{name}-10.png", f, -3, +3, -3, +3, n, n, colorspace=name)
    cplot.savefig(f"{name}-05.png", f, -3, +3, -3, +3, n, n, 0.5, name)
    cplot.savefig(f"{name}-00.png", f, -3, +3, -3, +3, n, n, 0, name)
