import matplotlib.pyplot as plt
import numpy as np
import sympy
from sympy import diff, lambdify, simplify
from sympy.abc import x

import cplot

fun = sympy.exp(x)
z0 = 0.0
text = "Taylor expansion of exp around 0"
p = cplot.Plotter((-5.0, 5.0, 400), (-5.0, 5.0, 400))

# fun = sympy.sin(x)
# z0 = 0.0
# text = "Taylor expansion of sin around 0"
# p = cplot.Plotter((-8.0, 8.0, 640), (-5.0, 5.0, 400))

# fun = sympy.log(x)
# z0 = 1.0
# text = "Taylor expansion of log around 1"
# p = cplot.Plotter((-0.7, 2.7, 400), (-1.7, 1.7, 400))

# fun = sympy.tan(x)
# z0 = 0.0
# text = "Taylor expansion of tan around 0"
# p = cplot.Plotter((-2.2, 2.2, 400), (-2.2, 2.2, 400))


z0 = np.full_like(p.Z, z0)
val = np.zeros_like(p.Z)

zk = np.ones_like(p.Z)

idx = 0
d = 1
val += lambdify(x, fun)(z0)
p.plot(val)
plt.suptitle(f"{text}, degree 0.00")
plt.savefig(f"data/out{idx:04d}.png", bbox_inches="tight")
plt.close()
idx += 1

# num intermediate steps
m = 20

dfun = fun
for k in range(1, 21):
    d *= k
    zk *= p.Z - z0
    dfun = diff(dfun, x)
    dfun = simplify(dfun)
    upd = lambdify(x, dfun)(z0) * zk / d
    for i in range(m):
        val += upd / m
        p.plot(val)
        plt.suptitle(f"{text}, degree {k - 1 + (i + 1) / m:.02f}")
        plt.savefig(f"data/out{idx:04d}.png", bbox_inches="tight")
        plt.close()
        idx += 1
