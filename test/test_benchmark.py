import cplot


def test_create():
    colorspace = "cielab"
    # colorspace = "cam16"
    # colorspace = "hsl"
    cplot.show_test_function("a", colorspace)
    cplot.show_test_function("b", colorspace)
    cplot.show_test_function("c", colorspace)


if __name__ == "__main__":
    test_create()
