from __future__ import annotations

from typing import Callable

import matplotlib.pyplot as plt
import numpy as np
import sympy
from rich.progress import track
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


def create_taylor_anim(taylor: Taylor, p: cplot.Plotter, name: str, max_degree: int):
    p.plot(lambdify(taylor.var, taylor.df)(p.Z))
    plt.savefig(f"{name}.svg", bbox_inches="tight")
    plt.close()

    idx = 0
    for k in track(range(max_degree + 1), description="Creating PNGs..."):
        val = next(taylor)
        p.plot(val)
        plt.suptitle(f"Taylor expansion of {name} around {taylor.z0}, degree {k}")
        plt.savefig(f"data/out{idx:04d}.png", bbox_inches="tight")
        plt.close()
        idx += 1


# name = "exp"
# p = cplot.Plotter((-7.0, 7.0, 400), (-7.0, 7.0, 400))
# taylor = Taylor(sympy.exp, 0, p.Z)
# max_degree = 30

# name = "sin"
# p = cplot.Plotter((-10.0, 10.0, 640), (-6.0, 6.0, 400))
# taylor = Taylor(sympy.sin, 0, p.Z)
# max_degree = 30

# name = "log"
# p = cplot.Plotter((-0.7, 2.7, 400), (-1.7, 1.7, 400))
# taylor = Taylor(sympy.log, 1, p.Z)
# max_degree = 40

name = "tan"
p = cplot.Plotter((-2.2, 2.2, 400), (-2.2, 2.2, 400))
taylor = Taylor(sympy.tan, 0, p.Z)
max_degree = 40


create_taylor_anim(taylor, p, name, max_degree)
