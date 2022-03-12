<p align="center">
  <a href="https://github.com/nschloe/cplot"><img alt="cplot" src="https://nschloe.github.io/cplot/cplot-logo.svg" width="50%"></a>
  <p align="center">Plot complex-valued functions with style.</p>
</p>

[![PyPi Version](https://img.shields.io/pypi/v/cplot.svg?style=flat-square)](https://pypi.org/project/cplot)
[![PyPI pyversions](https://img.shields.io/pypi/pyversions/cplot.svg?style=flat-square)](https://pypi.org/pypi/cplot/)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.5599493.svg?style=flat-square)](https://doi.org/10.5281/zenodo.5599493)
[![GitHub stars](https://img.shields.io/github/stars/nschloe/cplot.svg?style=flat-square&logo=github&label=Stars&logoColor=white)](https://github.com/nschloe/cplot)
[![Downloads](https://pepy.tech/badge/cplot/month)](https://pepy.tech/project/cplot)

[![Discord](https://img.shields.io/static/v1?logo=discord&label=chat&labelColor=white&message=on%20discord&color=7289da&style=flat-square)](https://discord.gg/hnTJ5MRX2Y)

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
    return np.sin(z**3) / z


plt = cplot.plot(
    f,
    (-2.0, +2.0, 400),
    (-2.0, +2.0, 400),
    # abs_scaling=lambda x: x / (x + 1),  # how to scale the lightness in domain coloring
    # contours_abs=2.0,
    # contours_arg=(-np.pi / 2, 0, np.pi / 2, np.pi),
    # emphasize_abs_contour_1: bool = True,
    # add_colorbars: bool = True,
    # add_axes_labels: bool = True,
    # saturation_adjustment: float = 1.28,
    # min_contour_length = None,
)
plt.show()
```

Historically, plotting of complex functions was in one of three ways

| <img src="https://nschloe.github.io/cplot/sinz3z-abs.svg" width="70%"> | <img src="https://nschloe.github.io/cplot/sinz3z-arg.svg" width="70%"> | <img src="https://nschloe.github.io/cplot/sinz3z-contours.svg" width="70%"> |
| :------------------------------------------------------------: | :------------------------------------------------------------: | :------------------------------------------------------------: |
|      Only show the absolute value; sometimes as a 3D plot         |        Only show the phase/the argument in a color wheel (phase portrait)        |          Show contour lines for both arg and abs                      |

Combining all three of them gives you a _cplot_:

<p align="center">
  <img src="https://nschloe.github.io/cplot/sinz3z.svg" width="60%">
</p>

See also [Wikipedia: Domain coloring](https://en.wikipedia.org/wiki/Domain_coloring).

Features of this software:

- cplot uses [OKLAB](https://bottosson.github.io/posts/oklab/), a perceptually
  uniform color space for the argument colors.
  This avoids streaks of colors occurring with other color spaces, e.g., HSL.
- The contour `abs(z) == 1` is emphasized, other abs contours are at 2, 4, 8, etc. and
  1/2, 1/4, 1/8, etc., respectively. This makes it easy to tell the absolte value
  precisely.
- For `arg(z) == 0`, the color is green, for `arg(z) == pi/2` it's blue, for `arg(z) =
  -pi / 2` it's orange, and for `arg(z) = pi` it's pink.

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

#### Riemann sphere

<p align="center">
  <img src="https://nschloe.github.io/cplot/riemann-log.png" width="60%">
</p>

cplot can also plot functions on the [Riemann
sphere](https://en.wikipedia.org/wiki/Riemann_sphere), a mapping of the complex
plane to the unit ball.

```python
import cplot
import numpy as np

cplot.riemann_sphere(np.log)
```

#### Gallery

[This way to the gallery!](https://github.com/nschloe/cplot/wiki/Gallery)

<a href="https://github.com/nschloe/cplot/wiki/Gallery">
  <img alt="cplot" src="https://nschloe.github.io/cplot/gallery-thumbnail.png" width="30%"/>
</a>


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
  Calendars _Complex Beauties_, 2011-](https://tu-freiberg.de/en/fakult1/ana/institute/institute-of-applied-analysis/organisation/complex-beauties)
- [Elias Wegert, _Visual Complex
  Functions_, 2012](https://www.springer.com/gp/book/9783034801799)
- [empet, _Visualizing complex-valued functions with Matplotlib and Mayavi, Domain coloring method_, 2014](https://nbviewer.org/github/empet/Math/blob/master/DomainColoring.ipynb)
- [John D. Cook, _Visualizing complex
  functions_, 2017](https://www.johndcook.com/blog/2017/11/09/visualizing-complex-functions/)
- [endolith, _complex-colormap_, 2017](https://github.com/endolith/complex_colormap)
- [Anthony Hernandez, _dcolor_, 2017](https://github.com/hernanat/dcolor)
- [Juan Carlos Ponce Campuzano, _DC
  gallery_, 2018](https://www.dynamicmath.xyz/domain-coloring/dcgallery.html)
- [3Blue1Brown, _Winding numbers and domain coloring_, 2018](https://youtu.be/b7FxPsqfkOY)
- [Ricky Reusser, _Domain Coloring with Adaptive
  Contouring_, 2019](https://observablehq.com/@rreusser/adaptive-domain-coloring)
- [Ricky Reusser, _Locally Scaled Domain Coloring, Part 1: Contour
  Plots_, 2020](https://observablehq.com/@rreusser/locally-scaled-domain-coloring-part-1-contour-plots)
- [David Lowry-Duda, _Visualizing modular forms_, 2020](https://arxiv.org/abs/2002.05234)

### License

This software is published under the [GPL-3.0 license](LICENSE). In cases where the
constraints of the GPL prevent you from using this software, feel free contact the
author.
