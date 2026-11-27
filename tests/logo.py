"""Update the color disk in assets/cplot-logo.svg with cplot's current colors.

The disk shows the phase wheel at |f| = 1, i.e., the colors of z / |z|, with positive
reals to the right.

    uv run tests/logo.py
"""

import base64
import io
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

import cplot

logo = Path(__file__).resolve().parent / ".." / "assets" / "cplot-logo.svg"


def create_disk(n: int = 338, supersample: int = 4) -> bytes:
    m = n * supersample
    x = (np.arange(m) + 0.5) / m * 2 - 1
    X, Y = np.meshgrid(x, -x)  # image rows go downwards; Im(z) upwards
    z = X + 1j * Y
    with np.errstate(invalid="ignore"):
        rgb = cplot.get_srgb1(z / np.abs(z))
    rgb = np.nan_to_num(rgb)
    alpha = (np.abs(z) <= 1.0).astype(float)
    rgba = np.dstack([rgb * alpha[..., None], alpha])  # premultiplied for averaging
    # box-filter downsampling for antialiasing
    rgba = rgba.reshape(n, supersample, n, supersample, 4).mean(axis=(1, 3))
    a = rgba[..., 3:]
    rgba[..., :3] = np.where(a > 0, rgba[..., :3] / np.maximum(a, 1e-12), 0.0)
    buf = io.BytesIO()
    plt.imsave(buf, np.clip(rgba, 0, 1), format="png")
    return buf.getvalue()


def update_logo():
    svg = logo.read_text()
    b64 = base64.b64encode(create_disk()).decode()
    svg, count = re.subn(
        r'(xlink:href="data:image/png;base64,)[^"]*(")', rf"\g<1>{b64}\g<2>", svg
    )
    assert count == 1
    logo.write_text(svg)


if __name__ == "__main__":
    update_logo()
