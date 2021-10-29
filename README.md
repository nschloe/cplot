<p align="center">
  <a href="https://github.com/nschloe/cplot"><img alt="cplot" src="https://nschloe.github.io/cplot/cplot-logo.svg" width="50%"></a>
  <p align="center">Plot complex-valued functions with style.</p>
</p>

[![PyPi Version](https://img.shields.io/pypi/v/cplot.svg?style=flat-square)](https://pypi.org/project/cplot)
[![PyPI pyversions](https://img.shields.io/pypi/pyversions/cplot.svg?style=flat-square)](https://pypi.org/pypi/cplot/)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.5599493.svg?style=flat-square)](https://doi.org/10.5281/zenodo.5599493)
[![GitHub stars](https://img.shields.io/github/stars/nschloe/cplot.svg?style=flat-square&logo=github&label=Stars&logoColor=white)](https://github.com/nschloe/cplot)
[![Downloads](https://pepy.tech/badge/cplot/month)](https://pepy.tech/project/cplot)

[![Discord](https://img.shields.io/static/v1?logo=discord&label=chat&message=on%20discord&color=7289da&style=flat-square)](https://discord.gg/hnTJ5MRX2Y)

[![gh-actions](https://img.shields.io/github/workflow/status/nschloe/cplot/ci?style=flat-square)](https://github.com/nschloe/cplot/actions?query=workflow%3Aci)
[![codecov](https://img.shields.io/codecov/c/github/nschloe/cplot.svg?style=flat-square)](https://codecov.io/gh/nschloe/cplot)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg?style=flat-square)](https://github.com/psf/black)

cplot helps plotting complex-valued functions in a visually appealing manner.

Install with

```
pip install cplot
```

and use as

```python
import numpy as np

import cplot


def f(z):
    return np.sin(z ** 3) / z


plt = cplot.plot(
    f,
    (-2.0, +2.0, 400),
    (-2.0, +2.0, 400),
    # abs_scaling=lambda x: x / (x + 1),  # how to scale the lightness in domain coloring
    # contours_abs="auto",
    # contours_arg=(-np.pi / 2, 0, np.pi / 2, np.pi),
    # emphasize_abs_contour_1: bool = True,
    # colorspace: str = "cam16",
    # add_colorbars: bool = True,
    # add_axes_labels: bool = True,
    # saturation_adjustment: float = 1.28,
)
plt.show()
```

<p align="center">
  <img src="https://nschloe.github.io/cplot/sinz3z.svg" width="50%">
</p>

The plot consists of three building blocks:

- [domain coloring](https://en.wikipedia.org/wiki/Domain_coloring), i.e.,
  mapping the absolute value to lightness and the complex argument to the chroma of
  the representing color
- Contours of constant absolute value (the contour `abs(z) == 1` is emphasized, the
  other contours are at (2, 4, 8, etc. and 1/2, 1/4, 1/8, etc., respectively)
- Contours along constant argument (angle). For `arg(z) == 0`, the color is green, for
  `arg(z) == pi/2` it's blue, for `arg(z) = -pi / 2` it's orange, and for `arg(z) = pi`
  it's pink

Other useful functions:

<!--pytest-codeblocks:skip-->

```python
# There is a tripcolor function as well for triangulated 2D domains
cplot.tripcolor(triang, z)

# The function get_srgb1 returns the SRGB1 triple for every complex input value.
# (Accepts arrays, too.)
z = 2 + 5j
val = cplot.get_srgb1(z)
```

<!--
- `abs_scaling` can be used to adjust the use of colors. `h` with a value less than
  `1.0` adds more color which can help isolating the roots and poles (which are still
  black and white, respectively). `h-0.0` ignores the magnitude of `f(z)` completely.
  `arctan` is another possible scaling.

- `colorspace` can be set to `hsl` to get the common fully saturated, vibrant colors.
  This is usually a bad idea since it creates artifacts which are not related with the
  underlying data. From [Wikipedia](https://en.wikipedia.org/wiki/Domain_coloring):

  > Since the HSL color space is not perceptually uniform, one can see streaks of
  > perceived brightness at yellow, cyan, and magenta (even though their absolute values
  > are the same as red, green, and blue) and a halo around L = 1/2. Use of the Lab
  > color space corrects this, making the images more accurate, but also makes them more
  > drab/pastel.

  Default is [`"cam16"`](https://doi.org/10.1002/col.22131);
  very similar is `"cielab"` (not shown here).
Consider the test function (math rendered with [xdoc](https://github.com/nschloe/xdoc))

```math
f(z) = \frac{(z^2 - 1) (z - 2 - 1j)^2}{z^2 + 2 + 2j}
```

|                               `h-1.0`                                |                               `h-0.5`                                |                               `h-0.0`                                |
| :------------------------------------------------------------------: | :------------------------------------------------------------------: | :------------------------------------------------------------------: |
| <img src="https://nschloe.github.io/cplot/cam16-10.svg" width="70%"> | <img src="https://nschloe.github.io/cplot/cam16-05.svg" width="70%"> | <img src="https://nschloe.github.io/cplot/cam16-00.svg" width="70%"> |
|  <img src="https://nschloe.github.io/cplot/hsl-10.svg" width="70%">  |  <img src="https://nschloe.github.io/cplot/hsl-05.svg" width="70%">  |  <img src="https://nschloe.github.io/cplot/hsl-00.svg" width="70%">  |

With this, it is easy to see where a function has very small and very large values, and
the multiplicty of zeros and poles is instantly identified by counting the color wheel
passes around a black or white point.
-->


### Testing

To run the cplot unit tests, check out this repository and run

```
tox
```

### Similar projects and further reading

- [Tristan Needham, _Visual Complex
  Analysis_, 1997](https://umv.science.upjs.sk/hutnik/NeedhamVCA.pdf)
- [François Labelle, _A Gallery of Complex
  Functions_, 2002](http://wismuth.com/complex/gallery.html)
- [Douglas Arnold and Jonathan Rogness, _Möbius transformations
  revealed_, 2008](https://youtu.be/0z1fIsUNhO4)
- [Konstantin Poelke and Konrad Polthier, _Lifted Domain Coloring_,
  2009](https://doi.org/10.1111/j.1467-8659.2009.01479.x)
- [Elias Wegert and Gunter Semmler, _Phase Plots of Complex Functions:
  a Journey in Illustration_, 2011](https://www.ams.org/notices/201106/rtx110600768p.pdf)
- [Elias Wegert,
  _Calendars "Complex Beauties_, 2011-](https://tu-freiberg.de/en/fakult1/ana/institute/institute-of-applied-analysis/organisation/complex-beauties)
- [Elias Wegert, _Visual Complex
  Functions_, 2012](https://www.springer.com/gp/book/9783034801799)
- [John D. Cook, _Visualizing complex
  functions_, 2017](https://www.johndcook.com/blog/2017/11/09/visualizing-complex-functions/)
- [endolith, _complex-colormap_, 2017](https://github.com/endolith/complex_colormap)
- [Anthony Hernandez, _dcolor_, 2017](https://github.com/hernanat/dcolor)
- [Juan Carlos Ponce Campuzano, _DC
  gallery_, 2018](https://www.dynamicmath.xyz/domain-coloring/dcgallery.html)
- [3Blue1Brown, _Winding numbers and domain coloring_, 2018](https://youtu.be/b7FxPsqfkOY)

### License

This software is published under the [GPL-3.0 license](LICENSE). In cases where the
constraints of the GPL prevent you from using this software, feel free contact the
author.
