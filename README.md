# cplot

Plotting complex-valued functions.

[![CircleCI](https://img.shields.io/circleci/project/github/nschloe/cplot/master.svg)](https://circleci.com/gh/nschloe/cplot/tree/master)
[![codecov](https://img.shields.io/codecov/c/github/nschloe/cplot.svg)](https://codecov.io/gh/nschloe/cplot)
[![PyPi Version](https://img.shields.io/pypi/v/cplot.svg)](https://pypi.python.org/pypi/cplot)
[![GitHub stars](https://img.shields.io/github/stars/nschloe/cplot.svg?logo=github&label=Stars&logoColor=white)](https://github.com/nschloe/cplot)

cplot is an attempt at encoding complex-valued data in colors. The general idea is to
map the absolute value to lightness and the complex argument (the "angle") to the
chroma.

```python
import cplot
import numpy

cplot.show(numpy.tan, -5, +5, -5, +5, 100, 100)
```

<img src="https://nschloe.github.io/cplot/z1.png" width="70%"> |
<img src="https://nschloe.github.io/cplot/z2.png" width="70%"> |
<img src="https://nschloe.github.io/cplot/z3.png" width="70%">
:-------------------:|:------------------:|:----------:|
`z**1`               |  `z**2`            |  `z**3`    |


<img src="https://nschloe.github.io/cplot/1z.png" width="70%"> |
<img src="https://nschloe.github.io/cplot/z+1-z-1.png" width="70%"> |
<img src="https://nschloe.github.io/cplot/z-1-z+1.png" width="70%">
:-------------------:|:------------------:|:-------------------:|
`1/z`                |  `(z+1) / (z-1)`   |  `(z-1) / (z+1)`    |

<img src="https://nschloe.github.io/cplot/sin.png" width="70%"> |
<img src="https://nschloe.github.io/cplot/cos.png" width="70%"> |
<img src="https://nschloe.github.io/cplot/tan.png" width="70%">
:-------------------:|:------------------:|:-------------------------:|
`numpy.sin`          |  `numpy.cos`       |  `scipy.tan`    |

<img src="https://nschloe.github.io/cplot/exp.png" width="70%"> |
<img src="https://nschloe.github.io/cplot/log.png" width="70%"> |
<img src="https://nschloe.github.io/cplot/gamma.png" width="70%">
:-------------------:|:------------------:|:-------------------------:|
`numpy.exp`          |  `numpy.log`       |  `scipy.special.gamma`    |

### Testing

To run the cplot unit tests, check out this repository and type
```
pytest
```

### Distribution

To create a new release

1. bump the `__version__` number,

2. tag and upload to PyPi:
    ```
    make publish
    ```

### License

cplot is published under the [MIT license](https://en.wikipedia.org/wiki/MIT_License).
