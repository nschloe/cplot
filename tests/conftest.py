import matplotlib
import pytest

# Never open windows during tests: render with the non-interactive Agg backend.
# This also covers plt.show() in the README code blocks (pytest --codeblocks).
matplotlib.use("Agg")


@pytest.fixture(autouse=True)
def _close_figures():
    yield
    import matplotlib.pyplot as plt

    plt.close("all")
