import matplotlib.pyplot as plt
import numpy as np
import pytest
from matplotlib.colors import to_rgb

import cplot
from cplot._colors import (
    _CORRECTION_T,
    _HUE_CORRECTION,
    _LIGHTNESS_CORRECTION,
    lightness_r,
    srgb1_grey,
)
from cplot._main import _contour_abs_colors

R = 10.0 ** np.linspace(-30, 30, 241)


def test_scale_abs_symmetry():
    s = cplot.scale_abs(R)
    assert np.all(np.diff(s) > 0)
    assert np.allclose(cplot.scale_abs(1 / R), 1 - s, atol=1e-15)


def test_scale_abs_special_values():
    s = cplot.scale_abs(np.array([0.0, 1.0, np.inf, np.nan]))
    assert s[0] == 0.0
    assert s[1] == 0.5
    assert s[2] == 1.0
    assert np.isnan(s[3])


def test_scale_abs_inv():
    assert np.allclose(cplot.scale_abs_inv(cplot.scale_abs(R)), R, rtol=1e-10)


def test_black_and_white_reserved():
    # 0 and infinity are exactly black and white; nothing in a wide range around
    # |f| = 1 is (in 8-bit sRGB)
    phi = np.exp(1j * np.linspace(-np.pi, np.pi, 64, endpoint=False))
    assert np.all(cplot.get_srgb1(np.array([0.0])) == 0.0)
    assert np.all(np.round(255 * cplot.get_srgb1(np.array([np.inf]))) == 255)
    for r in 10.0 ** np.arange(-100, 101, 10):
        rgb = np.round(255 * cplot.get_srgb1(r * phi))
        assert not np.any(np.all(rgb == 0, axis=-1))
        assert not np.any(np.all(rgb == 255, axis=-1))


def test_nan():
    assert np.all(np.isnan(cplot.get_srgb1(np.array([np.nan + 0j]))))


def test_red_at_one():
    r, g, b = cplot.get_srgb1(np.array([1.0 + 0j]))[0]
    assert r > g and r > b
    # the hue correction leaves arg 0 in place
    assert np.interp(0.0, _CORRECTION_T, _HUE_CORRECTION, period=2 * np.pi) == 0.0


def test_conventional_direction():
    # red -> yellow -> green -> cyan -> blue -> magenta when going counterclockwise
    rgb = cplot.get_srgb1(np.array([1, 1j, -1, -1j]))
    one, i, minus_one, minus_i = rgb
    assert i[1] > i[2]  # +i: yellowish (more green than blue)
    assert minus_one[0] < minus_one[1] and minus_one[0] < minus_one[2]  # -1: cyan
    assert minus_i[2] > minus_i[1]  # -i: blue-violet


def test_correction_tables():
    for table in (_HUE_CORRECTION, _LIGHTNESS_CORRECTION):
        assert table.shape == _CORRECTION_T.shape
        assert np.all(np.abs(table) < 0.15)


def test_grey_colorbar_is_grey():
    s = np.linspace(0, 1, 11)
    rgb = cplot.get_srgb1(np.ones(11), abs_scaling=lambda _: s, saturation_adjustment=0)
    assert np.allclose(rgb[:, 0], rgb[:, 1], atol=1e-3)
    assert np.allclose(rgb[:, 1], rgb[:, 2], atol=1e-3)
    assert np.allclose(rgb[:, 1], srgb1_grey(lightness_r(s)), atol=1e-3)


@pytest.mark.parametrize("alpha", [0.2, 0.4, 0.8])
def test_contour_contrast_is_constant(alpha):
    # grey line blended over the grey background at its level: lightness difference
    # is the same for all levels
    levels = 2.0 ** np.arange(-20, 21)
    sign = np.where(levels <= 1, 1.0, -1.0)
    line = np.array(
        [
            to_rgb(c)[0]
            for c in _contour_abs_colors(levels, cplot.scale_abs, alpha, sign)
        ]
    )
    lr_bg = lightness_r(cplot.scale_abs(levels))
    bg = srgb1_grey(lr_bg)
    mixed = (1 - alpha) * bg + alpha * line
    # invert srgb1_grey numerically
    grid = np.linspace(0, 1, 100001)
    lr_mixed = np.interp(mixed, srgb1_grey(grid), grid)
    contrast = (lr_mixed - lr_bg) * sign
    if alpha >= 0.4:
        assert np.allclose(contrast, 0.09, atol=0.002)
    else:
        # low opacity can't reach the target everywhere, but never flips
        assert np.all(contrast > 0)


def test_constant_function():
    cplot.plot(lambda z: 2.0, (-1, 1, 20), (-1, 1, 20))


def test_contours_abs_none():
    cplot.plot(np.sin, (-1, 1, 50), (-1, 1, 50), contours_abs=None, contours_arg=None)
    assert len(plt.gcf().axes[0].collections) == 0


def test_colorbar_labels_for_list():
    cplot.plot(np.sin, (-1, 1, 50), (-1, 1, 50), contours_abs=[0.5, 2])
    labels = [t.get_text() for t in plt.gcf().axes[1].get_yticklabels()]
    assert labels == ["0", "0.5", "2", "∞"]


def test_plot_phase_is_arg():
    assert cplot.plot_phase is cplot.plot_arg


def test_tricontour_abs_nan():
    import matplotlib.tri as tri

    x, y = np.meshgrid(np.linspace(-1, 1, 10), np.linspace(-1, 1, 10))
    x, y = x.ravel(), y.ravel()
    fz = x + 1j * y
    fz[0] = np.nan
    cplot.tricontour_abs(tri.Triangulation(x, y), fz)
