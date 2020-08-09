import cplot


def test_create():
    colorspace = "hsl"
    cplot.show_kovesi_test_image_radius(colorspace)
    cplot.show_kovesi_test_image_angle(colorspace)


if __name__ == "__main__":
    test_create()
