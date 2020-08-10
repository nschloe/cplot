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


def show_test_function(variant="a", colorspace="cam16", res=201):
    """Visual color map test after Peter Kovesi <https://arxiv.org/abs/1509.03700>,
    adapted for the complex color map.
    """
    if variant == "a":

        def f(z):
            r = numpy.abs(z)
            alpha = numpy.angle(z)
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
            x0 = numpy.arccos(-1 / (numpy.pi * k * w))
            x1 = 2 * numpy.pi - x0
            r1 = x1 / numpy.pi / k
            assert r1 + w * numpy.sin(k * numpy.pi * r1) >= 0
            return (r + w * numpy.sin(k * numpy.pi * r)) * numpy.exp(1j * alpha)

    elif variant == "b":

        def f(z):
            r = numpy.abs(z)
            alpha = numpy.angle(z)
            return r * numpy.exp(1j * (alpha + 0.8 * numpy.cos(3 * numpy.pi * alpha)))

    else:
        assert variant == "c"

        def f(z):
            return (z.real + 0.5 * numpy.sin(2 * numpy.pi * z.real)) + 1j * (
                z.imag + 0.5 * numpy.sin(2 * numpy.pi * z.imag)
            )

    show(f, -5, +5, -5, +5, res, res, colorspace=colorspace)
