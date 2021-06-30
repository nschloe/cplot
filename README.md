<p align="center">
  <a href="https://github.com/nschloe/cplot"><img alt="cplot" src="https://nschloe.github.io/cplot/cplot-logo.svg" width="50%"></a>
  <p align="center">Plot complex-valued functions with style.</p>
</p>

[![PyPi Version](https://img.shields.io/pypi/v/cplot.svg?style=flat-square)](https://pypi.org/project/cplot)
[![PyPI pyversions](https://img.shields.io/pypi/pyversions/cplot.svg?style=flat-square)](https://pypi.org/pypi/cplot/)
[![GitHub stars](https://img.shields.io/github/stars/nschloe/cplot.svg?style=flat-square&logo=github&label=Stars&logoColor=white)](https://github.com/nschloe/cplot)
[![PyPi downloads](https://img.shields.io/pypi/dm/cplot.svg?style=flat-square)](https://pypistats.org/packages/cplot)

[![Discord](https://img.shields.io/static/v1?logo=discord&label=chat&message=on%20discord&color=7289da&style=flat-square)](https://discord.gg/hnTJ5MRX2Y)

[![gh-actions](https://img.shields.io/github/workflow/status/nschloe/cplot/ci?style=flat-square)](https://github.com/nschloe/cplot/actions?query=workflow%3Aci)
[![codecov](https://img.shields.io/codecov/c/github/nschloe/cplot.svg?style=flat-square)](https://codecov.io/gh/nschloe/cplot)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg?style=flat-square)](https://github.com/psf/black)


cplot helps plotting complex-valued functions in a visually appealing manner.

There are two basic building blocks:

  * Plotting a contour map of the absolute value and/or the complex argument ("the
    angle")
  * Mapping the absolute value to lightness and the complex argument (the "angle") to
    the chroma of the representing color ([domain
    coloring](https://en.wikipedia.org/wiki/Domain_coloring))

Install with
```
pip install cplot
```
and use as
```python
import cplot
import numpy as np

cplot.show(np.tan, (-5, +5), (-5, +5), 100)

cplot.savefig("out.png", np.tan, (-5, +5), (-5, +5), 100)

# There is a tripcolor function as well for triangulated 2D domains
# cplot.tripcolor(triang, z)

# The function get_srgb1 returns the SRGB1 triple for every complex input value.
# (Accepts arrays, too.)
z = 2 + 5j
val = cplot.get_srgb1(z)
```
All functions have the optional arguments (with their default values)
```python
alpha = 1  # >= 0
colorspace = "cam16"  # "cielab", "oklab", "hsl"
```

* `alpha` can be used to adjust the use of colors. A value less than 1 adds more color
  which can help isolating the roots and poles (which are still black and white,
  respectively). `alpha=0` ignores the magnitude of `f(z)` completely.

* `colorspace` can be set to `hsl` to get the common fully saturated, vibrant
  colors. This is usually a bad idea since it creates artifacts which are not related
  with the underlying data. From [Wikipedia](https://en.wikipedia.org/wiki/Domain_coloring):

  > Since the HSL color space is not perceptually uniform, one can see streaks of
  > perceived brightness at yellow, cyan, and magenta (even though their absolute values
  > are the same as red, green, and blue) and a halo around L = 1/2. Use of the Lab
  > color space corrects this, making the images more accurate, but also makes them more
  > drab/pastel.

  Default is [`"cam16"`](http://onlinelibrary.wiley.com/doi/10.1002/col.22131/abstract);
  very similar is `"cielab"` (not shown here).

Consider the test function (math rendered with [xdoc](https://github.com/nschloe/xdoc))
```math
f(z) = \frac{(z^2 - 1) (z - 2 - 1j)^2}{z^2 + 2 + 2j}
```

| `alpha = 1`          |  `alpha = 0.5`       |  `alpha = 0.0`    |
| :----------:         |  :---------:         |  :--------:       |
| <img src="https://nschloe.github.io/cplot/cam16-10.png" width="70%"> | <img src="https://nschloe.github.io/cplot/cam16-05.png" width="70%"> | <img src="https://nschloe.github.io/cplot/cam16-00.png" width="70%"> |
| <img src="https://nschloe.github.io/cplot/hsl-10.png" width="70%"> | <img src="https://nschloe.github.io/cplot/hsl-05.png" width="70%"> | <img src="https://nschloe.github.io/cplot/hsl-00.png" width="70%"> |

The representation is chosen such that

  * values around **0** are **black**,
  * values around **infinity** are **white**,
  * values around **+1** are **green**,
  * values around **-1** are [**deep purple**](https://youtu.be/zUwEIt9ez7M),
  * values around **+i** are **blue**,
  * values around **-i** are **orange**.

(Compare to the z<sup>1</sup> reference plot below.)

With this, it is easy to see where a function has very small and very large values, and
the multiplicty of zeros and poles is instantly identified by counting the color wheel
passes around a black or white point.

#### Gallery

All plots are created with default settings.

<img src="https://nschloe.github.io/cplot/z1.png" width="70%"> | <img src="https://nschloe.github.io/cplot/z2.png" width="70%"> | <img src="https://nschloe.github.io/cplot/z3.png" width="70%">
:-------------------:|:------------------:|:----------:|
`z**1`               |  `z**2`            |  `z**3`    |

<img src="https://nschloe.github.io/cplot/1z.png" width="70%"> | <img src="https://nschloe.github.io/cplot/z-absz.png" width="70%"> | <img src="https://nschloe.github.io/cplot/z+1-z-1.png" width="70%"> |
:-------------------:|:------------------:|:----------:|
`1/z`               |  `z / abs(z)`            |  `(z+1) / (z-1)`    |

<img src="https://nschloe.github.io/cplot/zz.png" width="70%"> | <img src="https://nschloe.github.io/cplot/1zz.png" width="70%"> | <img src="https://nschloe.github.io/cplot/z1z.png" width="70%"> |
:-------------------:|:------------------:|:----------:|
`z ** z`               |  `(1/z) ** z`            |  `z ** (1/z)`    |

<img src="https://nschloe.github.io/cplot/root2.png" width="70%"> | <img src="https://nschloe.github.io/cplot/root3.png" width="70%"> | <img src="https://nschloe.github.io/cplot/root4.png" width="70%">
:-------------------:|:------------------:|:-------------------------:|
`np.sqrt`          |  `z**(1/3)`       |  `z**(1/4)`    |

<img src="https://nschloe.github.io/cplot/log.png" width="70%"> | <img src="https://nschloe.github.io/cplot/exp.png" width="70%"> | <img src="https://nschloe.github.io/cplot/exp1z.png" width="70%">
:-------------------:|:------------------:|:-------------------------:|
`np.log`          |  `np.exp`       |  `exp(1/z)`    |

<img src="https://nschloe.github.io/cplot/sin.png" width="70%"> | <img src="https://nschloe.github.io/cplot/cos.png" width="70%"> | <img src="https://nschloe.github.io/cplot/tan.png" width="70%">
:-------------------:|:------------------:|:-------------------------:|
`np.sin`          |  `np.cos`       |  `np.tan`    |

<img src="https://nschloe.github.io/cplot/sinh.png" width="70%"> | <img src="https://nschloe.github.io/cplot/cosh.png" width="70%"> | <img src="https://nschloe.github.io/cplot/tanh.png" width="70%">
:-------------------:|:------------------:|:-------------------------:|
`np.sinh`          |  `np.cosh`       |  `np.tanh`    |

<img src="https://nschloe.github.io/cplot/arcsin.png" width="70%"> | <img src="https://nschloe.github.io/cplot/arccos.png" width="70%"> | <img src="https://nschloe.github.io/cplot/arctan.png" width="70%">
:-------------------:|:------------------:|:-------------------------:|
`np.arcsin`          |  `np.arccos`       |  `np.arctan`    |

<img src="https://nschloe.github.io/cplot/sinz-z.png" width="70%"> | <img src="https://nschloe.github.io/cplot/cosz-z.png" width="70%"> | <img src="https://nschloe.github.io/cplot/tanz-z.png" width="70%">
:-------------------:|:------------------:|:----------------:|
`sin(z) / z`         |  `cos(z) / z`      |  `tan(z) / z`    |

<img src="https://nschloe.github.io/cplot/gamma.png" width="70%"> | <img src="https://nschloe.github.io/cplot/digamma.png" width="70%"> | <img src="https://nschloe.github.io/cplot/zeta.png" width="70%">
:-------------------:|:------------------:|:-------------------------:|
`scipy.special.gamma`          |  `scipy.special.digamma`       |  `mpmath.zeta`    |

<img src="https://nschloe.github.io/cplot/riemann-siegel-theta.png" width="70%"> | <img src="https://nschloe.github.io/cplot/riemann-siegel-z.png" width="70%"> | <img src="https://nschloe.github.io/cplot/riemann-xi.png" width="70%">
:-------------------:|:------------------:|:-------------------------:|
`mpmath.siegeltheta` |  `mpmath.siegelz`  |  Riemann-Xi    |


### Testing

To run the cplot unit tests, check out this repository and type
```
pytest
```

### Similar projects and further reading

  * https://github.com/endolith/complex_colormap
  * [John D.
    Cook](https://www.johndcook.com/blog/2017/11/09/visualizing-complex-functions/)
  * [Elias Wegert, Visual Complex
    Functions](https://www.springer.com/gp/book/9783034801799)
  * [Juan Carlos Ponce Campuzano, DC
    gallery](https://www.dynamicmath.xyz/domain-coloring/dcgallery.html)

### License
This software is published under the [GPLv3
license](https://www.gnu.org/licenses/gpl-3.0.en.html).
