import matplotlib.pyplot as plt
import numpy

from .main import show


def show_kovesi_test_image(cmap):
    """Visual color map test after Peter Kovesi <https://arxiv.org/abs/1509.03700>.
    """
    n = 300
    x = numpy.arange(n + 1) / n
    y = numpy.arange(n + 1) / n / 3
    X, Y = numpy.meshgrid(x, y)
    # From <https://arxiv.org/abs/1509.03700>:
    # It consists of a sine wave superimposed on a ramp function, this provides a set of
    # constant magnitude features presented at different offsets. The spatial frequency
    # of the sine wave is chosen to lie in the range at which the human eye is most
    # sensitive, and its amplitude is set so that the range from peak to trough
    # represents a series of features that are 10% of the total data range. The
    # amplitude of the sine wave is modulated from its full value at the top of the
    # image to zero at the bottom.
    Z = X + (3 * Y) ** 2 * 0.05 * numpy.sin(100 * numpy.pi * X)
    # Z = X + 0.05 * numpy.sin(100*numpy.pi*X*Y)

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


def show_kovesi_test_image_radius(colorspace="cam16"):
    """Visual color map test after Peter Kovesi <https://arxiv.org/abs/1509.03700>,
    adapted for the complex color map.
    """
    def f(z):
        r = numpy.abs(z)
        alpha = numpy.angle(z)
        return (r + 0.3 * numpy.sin(2 * numpy.pi * r)) * numpy.exp(1j * alpha)

    show(f, -5, +5, -5, +5, 101, 101, colorspace=colorspace)


def show_kovesi_test_image_angle(colorspace="cam16"):
    """Visual color map test after Peter Kovesi <https://arxiv.org/abs/1509.03700>,
    adapted for the complex color map.
    """
    def f(z):
        r = numpy.abs(z)
        alpha = numpy.angle(z)
        return r * numpy.exp(1j * (alpha + 0.3 * numpy.cos(3 * numpy.pi * alpha)))

    show(f, -5, +5, -5, +5, 101, 101, colorspace=colorspace)
