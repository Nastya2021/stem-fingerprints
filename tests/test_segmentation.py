import numpy as np
from methods.segmentation import otsu_threshold, otsu_binarize


def test_otsu_two_levels():
    img = np.array([[10, 10], [200, 200]], dtype=np.uint8)
    t = otsu_threshold(img)
    # будь-який поріг від 10 до 199 ділить класи однаково добре,
    # argmax бере перший, тобто 10
    assert t == 10


def test_otsu_binarize_two_levels():
    img = np.array([[10, 10], [200, 200]], dtype=np.uint8)
    expected = np.array([[0, 0], [255, 255]], dtype=np.uint8)
    assert np.array_equal(otsu_binarize(img), expected)


def test_otsu_uniform_image_no_crash():
    img = np.full((3, 3), 120, dtype=np.uint8)
    out = otsu_binarize(img)
    assert out.shape == (3, 3)
    assert set(np.unique(out)) <= {0, 255}

import numpy as np
from methods.segmentation import local_stats, sauvola_binarize


def test_local_stats_center_pixel():
    img = np.arange(1, 10, dtype=np.uint8).reshape(3, 3)
    mean, std = local_stats(img, size=3)
    assert mean[1, 1] == 5.0
    assert np.isclose(std[1, 1], np.sqrt(60 / 9))


def test_sauvola_uniform_is_white():
    img = np.full((5, 5), 100, dtype=np.uint8)
    out = sauvola_binarize(img, size=3)
    # std = 0, поріг = 100 * 0.8 = 80 < 100, тож усе фон
    assert np.array_equal(out, np.full((5, 5), 255, dtype=np.uint8))


def test_sauvola_finds_dark_line():
    img = np.full((5, 5), 200, dtype=np.uint8)
    img[:, 2] = 50
    out = sauvola_binarize(img, size=3)
    expected = np.full((5, 5), 255, dtype=np.uint8)
    expected[:, 2] = 0
    assert np.array_equal(out, expected)