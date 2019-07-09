# cplot

Plotting complex-valued functions.

[![CircleCI](https://img.shields.io/circleci/project/github/nschloe/cplot/master.svg?style=flat-square)](https://circleci.com/gh/nschloe/cplot/tree/master)
[![codecov](https://img.shields.io/codecov/c/github/nschloe/cplot.svg?style=flat-square)](https://codecov.io/gh/nschloe/cplot)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg?style=flat-square)](https://github.com/ambv/black)
[![PyPi Version](https://img.shields.io/pypi/v/cplot.svg?style=flat-square)](https://pypi.python.org/pypi/cplot)
[![GitHub stars](https://img.shields.io/github/stars/nschloe/cplot.svg?style=flat-square&logo=github&label=Stars&logoColor=white)](https://github.com/nschloe/cplot)

cplot is an attempt at encoding complex-valued data in colors. The general idea is to
map the absolute value to lightness and the complex argument (the "angle") to the chroma
of the representing color. This follows the [domain
coloring](https://en.wikipedia.org/wiki/Domain_coloring) approach with the colors taken
from the [CAM16](http://onlinelibrary.wiley.com/doi/10.1002/col.22131/abstract) to avoid
perceptual distortion.  (It has been claimed that this leads to drab images, but the
examples below prove the contrary.)

The representation is chosen such that

  * values around **0** are **black**,
  * values around **infinity** are **white**,
  * values around **+1** are **green**</span> ![green](https://nschloe.github.io/cplot/rgb_0_134_92.png "rgb(0,134,92)"),
  * values around **-1** are [**deep purple**](https://youtu.be/zUwEIt9ez7M) ![purple](https://nschloe.github.io/cplot/rgb_162_81_134.png "rgb(162,81,134)"),
  * values around **+i** are **blue** ![blue](https://nschloe.github.io/cplot/rgb_50_117_184.png "rgb(50,117,184)"), and
  * values around **-i** are **deep orange** ![orange](https://nschloe.github.io/cplot/rgb_155_101_0.png "rgb(155,101,0)").

(Compare to the z<sup>1</sup> reference plot below.)

With this, it is easy to see where a function has very small and very large values, and
the multiplicty of zeros and poles is instantly identified by counting the color wheel
passes around a black or white point.

See below for examples with some well-known functions.

```python
import cplot
import numpy

cplot.show(numpy.tan, -5, +5, -5, +5, 100, 100)

# There is a tripcolor function as well for triangulated 2D domains
# cplot.tripcolor(triang, z)

# The function get_srgb1 returns the SRGB1 triple for every complex input value.
# (Accepts arrays, too.)
z = 2 + 5j
val = cplot.get_srgb1(z)
```

<img src="https://nschloe.github.io/cplot/z1.png" width="70%"> |
<img src="https://nschloe.github.io/cplot/z2.png" width="70%"> |
<img src="https://nschloe.github.io/cplot/z3.png" width="70%">
:-------------------:|:------------------:|:----------:|
`z**1`               |  `z**2`            |  `z**3`    |

<img src="https://nschloe.github.io/cplot/1z.png" width="70%"> |
<img src="https://nschloe.github.io/cplot/z-absz.png" width="70%"> |
<img src="https://nschloe.github.io/cplot/z+1-z-1.png" width="70%"> |
:-------------------:|:------------------:|:----------:|
`1/z`               |  `z / abs(z)`            |  `(z+1) / (z-1)`    |

<img src="https://nschloe.github.io/cplot/root2.png" width="70%"> |
<img src="https://nschloe.github.io/cplot/root3.png" width="70%"> |
<img src="https://nschloe.github.io/cplot/root4.png" width="70%">
:-------------------:|:------------------:|:-------------------------:|
`numpy.sqrt`          |  `x**(1/3)`       |  `x**(1/4)`    |

<img src="https://nschloe.github.io/cplot/log.png" width="70%"> |
<img src="https://nschloe.github.io/cplot/exp.png" width="70%">
<img src="https://nschloe.github.io/cplot/exp1z.png" width="70%">
:-------------------:|:------------------:|:-------------------------:|
`numpy.log`          |  `numpy.exp`       |  `exp(1/x)`    |

<img src="https://nschloe.github.io/cplot/sin.png" width="70%"> |
<img src="https://nschloe.github.io/cplot/cos.png" width="70%"> |
<img src="https://nschloe.github.io/cplot/tan.png" width="70%">
:-------------------:|:------------------:|:-------------------------:|
`numpy.sin`          |  `numpy.cos`       |  `numpy.tan`    |

<img src="https://nschloe.github.io/cplot/sinh.png" width="70%"> |
<img src="https://nschloe.github.io/cplot/cosh.png" width="70%"> |
<img src="https://nschloe.github.io/cplot/tanh.png" width="70%">
:-------------------:|:------------------:|:-------------------------:|
`numpy.sinh`          |  `numpy.cosh`       |  `numpy.tanh`    |

<img src="https://nschloe.github.io/cplot/arcsin.png" width="70%"> |
<img src="https://nschloe.github.io/cplot/arccos.png" width="70%"> |
<img src="https://nschloe.github.io/cplot/arctan.png" width="70%">
:-------------------:|:------------------:|:-------------------------:|
`numpy.arcsin`          |  `numpy.arccos`       |  `numpy.arctan`    |

<img src="https://nschloe.github.io/cplot/gamma.png" width="70%"> |
<img src="https://nschloe.github.io/cplot/digamma.png" width="70%"> |
<img src="https://nschloe.github.io/cplot/zeta.png" width="70%">
:-------------------:|:------------------:|:-------------------------:|
`scipy.special.gamma`          |  `scipy.special.digamma`       |  `mpmath.zeta`    |


All functions have the optional parameter `alpha` (defaulting to `1`) which can be used
to tune the images. A value less than 1 makes gives adds more color which can help
isolating the roots and poles (which are still black and white, respectively). Consider
the function
```python
import cplot

def f(z):
  return (z ** 2 - 1) * (z - 2 - 1j) ** 2 / (z ** 2 + 2 + 2j)

cplot.savefig(
    "out.png", f, -3, +3, -3, +3, 200, 200,
    alpha=1.0
    # alpha=0.5
    # alpha=0.25
)
```

<img src="https://nschloe.github.io/cplot/f10.png" width="70%"> |
<img src="https://nschloe.github.io/cplot/f05.png" width="70%"> |
<img src="https://nschloe.github.io/cplot/f025.png" width="70%">
:-------------------:|:--------------------:|:------------------:|
`alpha = 1`          |  `alpha = 0.5`       |  `alpha = 0.25`    |


### Testing

To run the cplot unit tests, check out this repository and type
```
pytest
```

### License

cplot is published under the [MIT license](https://en.wikipedia.org/wiki/MIT_License).
