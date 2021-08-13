import matplotlib.pyplot as plt
import numpy as np

from ._main import plot_colors


def show_kovesi_test_image(cmap):
    """Visual color map test after Peter Kovesi <https://arxiv.org/abs/1509.03700>."""
    n = 300
    x = np.arange(n + 1) / n
    y = np.arange(n + 1) / n / 3
    X, Y = np.meshgrid(x, y)
    # From <https://arxiv.org/abs/1509.03700>:
    # It consists of a sine wave superimposed on a ramp function, this provides a set of
    # constant magnitude features presented at different offsets. The spatial frequency
    # of the sine wave is chosen to lie in the range at which the human eye is most
    # sensitive, and its amplitude is set so that the range from peak to trough
    # represents a series of features that are 10% of the total data range. The
    # amplitude of the sine wave is modulated from its full value at the top of the
    # image to zero at the bottom.
    Z = X + (3 * Y) ** 2 * 0.05 * np.sin(100 * np.pi * X)
    # Z = X + 0.05 * np.sin(100*np.pi*X*Y)

    plt.imshow(
        Z,
        extent=(x.min(), x.max(), y.max(), y.min()),
        interpolation="nearest",
        cmap=cmap,
        origin="lower",
        aspect="equal",
    )

    plt.xticks([])
    plt.yticks([])

    plt.show()


def show_test_function(variant="a", colorspace="cam16", res=201):
    """Visual color map test after Peter Kovesi <https://arxiv.org/abs/1509.03700>,
    adapted for the complex color map.
    """

    def fa(z):
        r = np.abs(z)
        alpha = np.angle(z)
        # for the radius function
        #
        #   f(r) = r + w * sin(k * pi * r)
        #
        # to be >= 0 everwhere, the first minimum at
        #
        #  r0 = arccos(-1 / (pi * k * w))
        #  r1 = 1 / (pi * k) (pi + (pi - y))
        #     = (2 pi - y) / (pi * k)
        #
        # has to be >= 0, i.e.,
        #
        k = 2
        w = 0.7
        x0 = np.arccos(-1 / (np.pi * k * w))
        x1 = 2 * np.pi - x0
        r1 = x1 / np.pi / k
        assert r1 + w * np.sin(k * np.pi * r1) >= 0
        return (r + w * np.sin(k * np.pi * r)) * np.exp(1j * alpha)

    def fb(z):
        r = np.abs(z)
        alpha = np.angle(z)
        return r * np.exp(1j * (alpha + 0.8 * np.cos(3 * np.pi * alpha)))

    def fc(z):
        return (z.real + 0.5 * np.sin(2 * np.pi * z.real)) + 1j * (
            z.imag + 0.5 * np.sin(2 * np.pi * z.imag)
        )

    f = {"a": fa, "b": fb, "c": fc}[variant]

    plot_colors(f, (-5, +5), (-5, +5), res, colorspace=colorspace)
    plt.show()
