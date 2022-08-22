import matplotlib
import matplotlib.pyplot as plt
import numpy as np

import cplot

# https://github.com/matplotlib/matplotlib/issues/23701#issuecomment-1222008929
matplotlib.use("GTK3Agg")

p = cplot.Plotter((-2.2, 2.2, 400), (-2.2, 2.2, 400))

for idx, a in enumerate(np.linspace(-5.0, 5.0, 501)):
    p.plot(p.Z**a)
    plt.suptitle(f"$z^{{{a:.2f}}}$")
    plt.savefig(f"data/out{idx:04d}.png", bbox_inches="tight")
    plt.close()
