from __future__ import annotations

from typing import Callable

import matplotlib.pyplot as plt
import numpy as np
import sympy
from sympy import diff, lambdify, simplify

import cplot


class Taylor:
    def __init__(self, f: Callable, z0: complex, Z: np.ndarray):
        self.z0 = z0
        self.var = sympy.Symbol("z")
        self.Zz0 = Z - self.z0
        self.val = None
        self.zk = np.ones_like(Z)
        self.k = 0
        self.df = f(self.var)

    def __next__(self) -> np.ndarray:
        if self.k == 0:
            self.val = lambdify(self.var, self.df)(self.z0).astype(complex) * self.zk
            self.k += 1
            return self.val

        self.zk *= self.Zz0 / self.k
        self.df = simplify(diff(self.df, self.var))
        self.val += lambdify(self.var, self.df)(self.z0) * self.zk
        self.k += 1
        return self.val


def create_taylor_anim(taylor, p, title, max_degree):
    idx = 0
    for k in range(max_degree + 1):
        print(f"{k}...")
        val = next(taylor)
        p.plot(val)
        plt.suptitle(f"{title}, degree {k}")
        plt.savefig(f"data/out{idx:04d}.png", bbox_inches="tight")
        plt.close()
        idx += 1


# title = "Taylor expansion of exp around 0"
# p = cplot.Plotter((-7.0, 7.0, 400), (-7.0, 7.0, 400))
# taylor = Taylor(sympy.exp, 0.0, p.Z)
# max_degree = 30

# title = "Taylor expansion of sin around 0"
# p = cplot.Plotter((-10.0, 10.0, 640), (-6.0, 6.0, 400))
# taylor = Taylor(sympy.sin, 0.0, p.Z)
# max_degree = 30

# title = "Taylor expansion of log around 1"
# p = cplot.Plotter((-0.7, 2.7, 400), (-1.7, 1.7, 400))
# taylor = Taylor(sympy.log, 1.0, p.Z)
# max_degree = 40

title = "Taylor expansion of tan around 0"
p = cplot.Plotter((-2.2, 2.2, 400), (-2.2, 2.2, 400))
taylor = Taylor(sympy.tan, 0.0, p.Z)
max_degree = 40


create_taylor_anim(taylor, p, title, max_degree)
