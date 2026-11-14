import numpy as np
from methods.morphology import erode, dilate, opening, closing


def _block():
    img = np.full((5, 5), 255, dtype=np.uint8)
    img[1:4, 1:4] = 0          # квадрат 3x3 гребеня
    return img


def test_erode_shrinks_block_to_center():
    out = erode(_block(), size=3)
    expected = np.full((5, 5), 255, dtype=np.uint8)
    expected[2, 2] = 0         # лишився тільки центр
    assert np.array_equal(out, expected)


def test_dilate_single_pixel_grows_to_3x3():
    img = np.full((5, 5), 255, dtype=np.uint8)
    img[2, 2] = 0
    out = dilate(img, size=3)
    expected = np.full((5, 5), 255, dtype=np.uint8)
    expected[1:4, 1:4] = 0
    assert np.array_equal(out, expected)


def test_opening_removes_isolated_dot():
    img = np.full((5, 5), 255, dtype=np.uint8)
    img[2, 2] = 0              # одинока крапка
    assert np.array_equal(opening(img, size=3), np.full((5, 5), 255, dtype=np.uint8))


def test_closing_fills_small_gap():
    img = np.full((5, 5), 255, dtype=np.uint8)
    img[2, 0:2] = 0
    img[2, 3:5] = 0            # розрив в одну клітинку посередині
    out = closing(img, size=3)
    assert out[2, 2] == 0      # розрив заповнено