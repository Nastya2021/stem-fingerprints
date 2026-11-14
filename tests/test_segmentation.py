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