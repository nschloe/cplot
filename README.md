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
    # linewidth = None,
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

<!--pytest.mark.skip-->

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

<!--pytest-codeblocks:importorskip(pyvista)-->

```python
import cplot
import numpy as np

cplot.riemann_sphere(np.log)
```

#### Gallery

All plots are created with default settings.

| <img src="https://nschloe.github.io/cplot/z1.png" width="70%"> | <img src="https://nschloe.github.io/cplot/z2.png" width="70%"> | <img src="https://nschloe.github.io/cplot/z3.png" width="70%"> |
| :------------------------------------------------------------: | :------------------------------------------------------------: | :------------------------------------------------------------: |
|                            `z ** 1`                            |                            `z ** 2`                            |                            `z ** 3`                            |

<details>
<summary>Many more plots</summary>

| <img src="https://nschloe.github.io/cplot/1z.png" width="70%"> | <img src="https://nschloe.github.io/cplot/1z2.png" width="70%"> | <img src="https://nschloe.github.io/cplot/1z3.png" width="70%"> |
| :------------------------------------------------------------: | :------------------------------------------------------------: | :------------------------------------------------------------: |
|                            `1 / z`                            |                            `1 / z ** 2`                            |                            `1 / z ** 3`                            |


| <img src="https://nschloe.github.io/cplot/moebius1.png" width="70%"> | <img src="https://nschloe.github.io/cplot/moebius2.png" width="70%"> | <img src="https://nschloe.github.io/cplot/moebius3.png" width="70%"> |
| :------------------------------------------------------------: | :-------------------------------------------------------------: | :-----------------------------------------------------------------: |
|                            `(z + 1) / (z - 1)`                             |                         Another [Möbius transformation](https://en.wikipedia.org/wiki/M%C3%B6bius_transformation)                           |                       A third Möbius transformation                        |

| <img src="https://nschloe.github.io/cplot/re.png" width="70%"> | <img src="https://nschloe.github.io/cplot/z-absz.png" width="70%"> | <img src="https://nschloe.github.io/cplot/conj.png" width="70%"> |
| :------------------------------------------------------------: | :----------------------------------------------------------------: | :--------------------------------------------------------------: |
|                           `np.real`                            |                            `z / abs(z)`                            |                            `np.conj`                             |

| <img src="https://nschloe.github.io/cplot/z6+1.png" width="70%"> | <img src="https://nschloe.github.io/cplot/z6-1.png" width="70%"> | <img src="https://nschloe.github.io/cplot/z-6+1.png" width="70%"> |
| :--------------------------------------------------------------: | :--------------------------------------------------------------: | :---------------------------------------------------------------: |
|                           `z ** 6 + 1`                           |                           [`z ** 6 - 1`](https://en.wikipedia.org/wiki/Root_of_unity)                           |                          `z ** (-6) + 1`                          |

| <img src="https://nschloe.github.io/cplot/zz.png" width="70%"> | <img src="https://nschloe.github.io/cplot/1zz.png" width="70%"> | <img src="https://nschloe.github.io/cplot/z1z.png" width="70%"> |
| :------------------------------------------------------------: | :-------------------------------------------------------------: | :-------------------------------------------------------------: |
|                            `z ** z`                            |                          `(1/z) ** z`                           |                          `z ** (1/z)`                           |

| <img src="https://nschloe.github.io/cplot/root2.png" width="70%"> | <img src="https://nschloe.github.io/cplot/root3.png" width="70%"> | <img src="https://nschloe.github.io/cplot/root4.png" width="70%"> |
| :---------------------------------------------------------------: | :---------------------------------------------------------------: | :---------------------------------------------------------------: |
|                             `np.sqrt`                             |                            `z**(1/3)`                             |                            `z**(1/4)`                             |

| <img src="https://nschloe.github.io/cplot/log.png" width="70%"> | <img src="https://nschloe.github.io/cplot/exp.png" width="70%"> | <img src="https://nschloe.github.io/cplot/exp2.png" width="70%"> |
| :-------------------------------------------------------------: | :-------------------------------------------------------------: | :--------------------------------------------------------------: |
|                            [`np.log`](https://en.wikipedia.org/wiki/Logarithm)                             |                            `np.exp`                             |                            `np.exp2`                             |

| <img src="https://nschloe.github.io/cplot/exp1z.png" width="70%"> | <img src="https://nschloe.github.io/cplot/zsin1z.png" width="70%"> | <img src="https://nschloe.github.io/cplot/cos1z.png" width="70%"> |
| :---------------------------------------------------------------: | :----------------------------------------------------------------: | :---------------------------------------------------------------: |
|                          `np.exp(1 / z)`                          |                        `z * np.sin(1 / z)`                         |                          `np.cos(1 / z)`                          |

| <img src="https://nschloe.github.io/cplot/exp-z2.png" width="70%"> | <img src="https://nschloe.github.io/cplot/11z2.png" width="70%"> | <img src="https://nschloe.github.io/cplot/erf.png" width="70%"> |
| :----------------------------------------------------------------: | :--------------------------------------------------------------: | :-------------------------------------------------------------: |
|                          `exp(- z ** 2)`                           |                        [`1 / (1 + z ** 2)`](https://en.wikipedia.org/wiki/Runge%27s_phenomenon)                        |                       [Error function](https://en.wikipedia.org/wiki/Error_function)                       |

| <img src="https://nschloe.github.io/cplot/sin.png" width="70%"> | <img src="https://nschloe.github.io/cplot/cos.png" width="70%"> | <img src="https://nschloe.github.io/cplot/tan.png" width="70%"> |
| :-------------------------------------------------------------: | :-------------------------------------------------------------: | :-------------------------------------------------------------: |
|                            `np.sin`                             |                            `np.cos`                             |                            `np.tan`                             |

| <img src="https://nschloe.github.io/cplot/sec.png" width="70%"> | <img src="https://nschloe.github.io/cplot/csc.png" width="70%"> | <img src="https://nschloe.github.io/cplot/cot.png" width="70%"> |
| :-------------------------------------------------------------: | :-------------------------------------------------------------: | :-------------------------------------------------------------: |
|                              `sec`                              |                              `csc`                              |                              `cot`                              |

| <img src="https://nschloe.github.io/cplot/sinh.png" width="70%"> | <img src="https://nschloe.github.io/cplot/cosh.png" width="70%"> | <img src="https://nschloe.github.io/cplot/tanh.png" width="70%"> |
| :--------------------------------------------------------------: | :--------------------------------------------------------------: | :--------------------------------------------------------------: |
|                            [`np.sinh`](https://en.wikipedia.org/wiki/Hyperbolic_functions)                             |                            `np.cosh`                             |                            `np.tanh`                             |

| <img src="https://nschloe.github.io/cplot/sech.png" width="70%"> | <img src="https://nschloe.github.io/cplot/csch.png" width="70%"> | <img src="https://nschloe.github.io/cplot/coth.png" width="70%"> |
| :--------------------------------------------------------------: | :--------------------------------------------------------------: | :--------------------------------------------------------------: |
|                            secans hyperbolicus                             |                            cosecans hyperbolicus                             |                           cotangent hyperbolicus                            |


| <img src="https://nschloe.github.io/cplot/arcsin.png" width="70%"> | <img src="https://nschloe.github.io/cplot/arccos.png" width="70%"> | <img src="https://nschloe.github.io/cplot/arctan.png" width="70%"> |
| :----------------------------------------------------------------: | :----------------------------------------------------------------: | :----------------------------------------------------------------: |
|                            `np.arcsin`                             |                            `np.arccos`                             |                            `np.arctan`                             |

| <img src="https://nschloe.github.io/cplot/sinz-z.png" width="70%"> | <img src="https://nschloe.github.io/cplot/cosz-z.png" width="70%"> | <img src="https://nschloe.github.io/cplot/tanz-z.png" width="70%"> |
| :----------------------------------------------------------------: | :----------------------------------------------------------------: | :----------------------------------------------------------------: |
|                            [Sinc, `sin(z) / z`](https://en.wikipedia.org/wiki/Sinc_function)                            |                            `cos(z) / z`                            |                            `tan(z) / z`                            |

| <img src="https://nschloe.github.io/cplot/si.png" width="70%"> | <img src="https://nschloe.github.io/cplot/ci.png" width="70%"> | <img src="https://nschloe.github.io/cplot/lambertw.png" width="70%"> |
| :----------------------------------------------------------------: | :----------------------------------------------------------------: | :----------------------------------------------------------------: |
|              [Integral sine _Si_](https://en.wikipedia.org/wiki/Trigonometric_integral)                      |         Integral cosine _Ci_                         |                            [Lambert W function](https://en.wikipedia.org/wiki/Lambert_W_function)           |


| <img src="https://nschloe.github.io/cplot/zeta.png" width="70%"> | <img src="https://nschloe.github.io/cplot/bernoulli.png" width="70%"> | <img src="https://nschloe.github.io/cplot/dirichlet-eta.png" width="70%"> |
| :---------------------------------------------------------------: | :-----------------------------------------------------------------: | :--------------------------------------------------------------: |
|                      [`mpmath.zeta`](https://en.wikipedia.org/wiki/Riemann_zeta_function)                       |                       Bernoulli function         |    [Dirichlet eta function](https://en.wikipedia.org/wiki/Dirichlet_eta_function)    |

| <img src="https://nschloe.github.io/cplot/hurwitz-zeta-1-3.png" width="70%"> | <img src="https://nschloe.github.io/cplot/hurwitz-zeta-24-25.png" width="70%"> | <img src="https://nschloe.github.io/cplot/hurwitz-zeta-3-4i.png" width="70%"> |
| :---------------------------------------------------------------: | :-----------------------------------------------------------------: | :--------------------------------------------------------------: |
|                       [Hurwitz zeta function](https://en.wikipedia.org/wiki/Hurwitz_zeta_function) with `a = 1/3`         |                          Hurwitz zeta function with `a = 24/25`                   |                          Hurwitz zeta function with `a = 3 + 4i`                |


| <img src="https://nschloe.github.io/cplot/gamma.png" width="70%"> | <img src="https://nschloe.github.io/cplot/reciprocal-gamma.png" width="70%"> | <img src="https://nschloe.github.io/cplot/digamma.png" width="70%"> |
| :---------------------------------------------------------------: | :-----------------------------------------------------------------: | :--------------------------------------------------------------: |
|                       [`scipy.special.gamma`](https://en.wikipedia.org/wiki/Gamma_function)                       |                       [reciprocal Gamma](https://en.wikipedia.org/wiki/Reciprocal_gamma_function)                       |                          [`scipy.special.digamma`](https://en.wikipedia.org/wiki/Digamma_function)                      |

| <img src="https://nschloe.github.io/cplot/riemann-siegel-theta.png" width="70%"> | <img src="https://nschloe.github.io/cplot/riemann-siegel-z.png" width="70%"> | <img src="https://nschloe.github.io/cplot/riemann-xi.png" width="70%"> |
| :------------------------------------------------------------------------------: | :--------------------------------------------------------------------------: | :--------------------------------------------------------------------: |
|                               [Riemann-Siegel theta function](https://en.wikipedia.org/wiki/Riemann%E2%80%93Siegel_theta_function)                               |                               [Z-function](https://en.wikipedia.org/wiki/Z_function)                               |                               [Riemann-Xi](https://en.wikipedia.org/wiki/Riemann_Xi_function)                               |

| <img src="https://nschloe.github.io/cplot/ellipj-sn-06.png" width="70%"> | <img src="https://nschloe.github.io/cplot/ellipj-cn-06.png" width="70%"> | <img src="https://nschloe.github.io/cplot/ellipj-dn-06.png" width="70%"> |
| :----------------------------------------------------------------------: | :----------------------------------------------------------------------: | :----------------------------------------------------------------------: |
|                    [Jacobi elliptic function](https://en.wikipedia.org/wiki/Jacobi_elliptic_functions) `sn(0.6)`                    |                                `cn(0.6)`                                 |                                `dn(0.6)`                                 |


| <img src="https://nschloe.github.io/cplot/jtheta1.png" width="70%"> | <img src="https://nschloe.github.io/cplot/jtheta2.png" width="70%"> | <img src="https://nschloe.github.io/cplot/jtheta3.png" width="70%"> |
| :----------------------------------------------------------------------: | :----------------------------------------------------------------------: | :----------------------------------------------------------------------: |
|                    [Jacobi theta](https://en.wikipedia.org/wiki/Theta_function) 1 with `q=0.1 * exp(0.1j * np.pi))`                    |      Jacobi theta 2 with the same `q`                                 |                              Jacobi theta 3 with the same `q`                             |

| <img src="https://nschloe.github.io/cplot/bessel1-1.png" width="70%"> | <img src="https://nschloe.github.io/cplot/bessel1-2.png" width="70%"> | <img src="https://nschloe.github.io/cplot/bessel1-3.png" width="70%"> |
| :------------------------------------------------------------------: | :------------------------------------------------------------------: | :------------------------------------------------------------------: |
|                 [Bessel function](https://en.wikipedia.org/wiki/Bessel_function), first kind, order 1                 |                               Bessel function, first kind, order 2                                |                              Bessel function, first kind, order 3                                |

| <img src="https://nschloe.github.io/cplot/bessel2-1.png" width="70%"> | <img src="https://nschloe.github.io/cplot/bessel2-2.png" width="70%"> | <img src="https://nschloe.github.io/cplot/bessel2-3.png" width="70%"> |
| :------------------------------------------------------------------: | :------------------------------------------------------------------: | :------------------------------------------------------------------: |
|                 Bessel function, second kind, order 1                 |                              Bessel function, second kind, order 2                                |                              Bessel function, second kind, order 3                                |

| <img src="https://nschloe.github.io/cplot/hankel1a.png" width="70%"> | <img src="https://nschloe.github.io/cplot/hankel1b.png" width="70%"> | <img src="https://nschloe.github.io/cplot/hankel2.png" width="70%"> |
| :------------------------------------------------------------: | :------------------------------------------------------------: | :------------------------------------------------------------: |
|                           Hankel function of first kind (n=1.0)                           |                           Hankel function of first kind (n=3.1)                           |                          Hankel function of second kind (n=1.0)                           |

| <img src="https://nschloe.github.io/cplot/fresnel-s.png" width="70%"> | <img src="https://nschloe.github.io/cplot/fresnel-c.png" width="70%"> | <img src="https://nschloe.github.io/cplot/faddeeva.png" width="70%"> |
| :-------------------------------------------------------------------: | :-------------------------------------------------------------------: | :------------------------------------------------------------------: |
|                               [Fresnel S](https://en.wikipedia.org/wiki/Fresnel_integral)                               |                               [Fresnel C](https://en.wikipedia.org/wiki/Fresnel_integral)                               |                          [Faddeeva function](https://en.wikipedia.org/wiki/Faddeeva_function)                           |

| <img src="https://nschloe.github.io/cplot/airy-ai.png" width="70%"> | <img src="https://nschloe.github.io/cplot/airy-bi.png" width="70%"> | <img src="https://nschloe.github.io/cplot/airye-ai.png" width="70%"> |
| :-----------------------------------------------------------------: | :-----------------------------------------------------------------: | :------------------------------------------------------------------: |
|                          [Airy function Ai](https://en.wikipedia.org/wiki/Airy_function)                           |                                 [Bi](https://en.wikipedia.org/wiki/Airy_function)                                  |                       [Exponentially scaled eAi](https://en.wikipedia.org/wiki/Airy_function)                       |

| <img src="https://nschloe.github.io/cplot/tanh-sinh.png" width="70%"> | <img src="https://nschloe.github.io/cplot/sinh-sinh.png" width="70%"> | <img src="https://nschloe.github.io/cplot/exp-sinh.png" width="70%"> |
| :-------------------------------------------------------------------: | :-------------------------------------------------------------------: | :------------------------------------------------------------------: |
|                       `tanh(pi / 2 * sinh(z))`                        |                       `sinh(pi / 2 * sinh(z))`                        |                       `exp(pi / 2 * sinh(z))`                        |


| <img src="https://nschloe.github.io/cplot/kleinj.png" width="70%"> | <img src="https://nschloe.github.io/cplot/dedekind-eta.png" width="70%"> |
| :-------------------------------------------------------------------: | :-------------------------------------------------------------------: |
|         [Klein's _j_-invariant](https://en.wikipedia.org/wiki/J-invariant)                     |                 [Dedekind eta function](https://en.wikipedia.org/wiki/Dedekind_eta_function)                      |

| <img src="https://nschloe.github.io/cplot/lambert-1.png" width="70%"> | <img src="https://nschloe.github.io/cplot/lambert-von-mangoldt.png" width="70%"> | <img src="https://nschloe.github.io/cplot/lambert-liouville.png" width="70%"> |
| :-------------------------------------------------------------------: | :-------------------------------------------------------------------: | :------------------------------------------------------------------: |
|                      [Lambert series](https://en.wikipedia.org/wiki/Lambert_series) with 1s                        |                       Lambert series with von-Mangoldt-coefficients                        |                      Lambert series with Liouville-coefficients                        |
</details>

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
