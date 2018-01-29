# cplot

Plotting complex-valued functions.

[![CircleCI](https://img.shields.io/circleci/project/github/nschloe/cplot/master.svg)](https://circleci.com/gh/nschloe/cplot/tree/master)
[![codecov](https://img.shields.io/codecov/c/github/nschloe/cplot.svg)](https://codecov.io/gh/nschloe/cplot)
[![awesome](https://img.shields.io/badge/awesome-yes-brightgreen.svg)](https://img.shields.io/badge/awesome-yes-brightgreen.svg)
[![PyPi Version](https://img.shields.io/pypi/v/cplot.svg)](https://pypi.python.org/pypi/cplot)
[![GitHub stars](https://img.shields.io/github/stars/nschloe/cplot.svg?style=social&label=Stars)](https://github.com/nschloe/cplot)


```python
import cplot
import numpy

cplot.show(numpy.tan, -5, +5, -5, +5, 100, 100)
```
produces
<img src="https://nschloe.github.io/cplot/tan.png" width="30%">

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
