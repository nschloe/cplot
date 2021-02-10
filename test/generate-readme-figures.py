import mpmath
import numpy as np
import scipy.special

import cplot


def zeta(z):
    vals = [[mpmath.zeta(val) for val in row] for row in z]
    out = np.array(
        [[float(val.real) + 1j * float(val.imag) for val in row] for row in vals]
    )
    return out


# First function from the SIAM-100-digit challenge
# <https://en.wikipedia.org/wiki/Hundred-dollar,_Hundred-digit_Challenge_problems>
def siam(z):
    return np.cos(np.log(z) / z) / z


n = 400

cplot.imsave("z1.png", lambda z: z ** 1, -2, +2, -2, +2, n, n)
cplot.imsave("z2.png", lambda z: z ** 2, -2, +2, -2, +2, n, n)
cplot.imsave("z3.png", lambda z: z ** 3, -2, +2, -2, +2, n, n)

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
cplot.savefig("zeta.png", zeta, -30, +30, -30, +30, n, n)

cplot.savefig("siam.png", siam, -1, 1, -1, 1, n, n, alpha=0.1)
