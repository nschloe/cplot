import matplotlib
import matplotlib.pyplot as plt

# Never open windows during tests: render with the non-interactive Agg backend.
# This also covers plt.show() in the README code blocks (pytest --codeblocks).
matplotlib.use("Agg")


def pytest_runtest_teardown(item):
    # Close all figures after every test, including the README code blocks (which are
    # not regular test functions, so an autouse fixture wouldn't apply). Otherwise a
    # test may draw into the figure left over by another one.
    plt.close("all")
