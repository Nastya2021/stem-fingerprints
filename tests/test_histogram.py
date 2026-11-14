import numpy as np
from methods.histogram import equalize_histogram


def test_equalize_simple():
    img = np.array([[50, 50], [100, 150]], dtype=np.uint8)
    expected = np.array([[0, 0], [128, 255]], dtype=np.uint8)
    assert np.array_equal(equalize_histogram(img), expected)


def test_equalize_uniform_image():
    img = np.full((3, 3), 77, dtype=np.uint8)
    assert np.array_equal(equalize_histogram(img), img)


def test_equalize_stretches_range():
    img = np.array([[100, 110], [120, 130]], dtype=np.uint8)
    out = equalize_histogram(img)
    assert out.min() == 0
    assert out.max() == 255