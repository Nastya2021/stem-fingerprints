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
    
from methods.morphology import thin


def test_thin_single_pixel_stays():
    img = np.full((5, 5), 255, dtype=np.uint8)
    img[2, 2] = 0
    assert np.array_equal(thin(img), img)


def test_thin_one_pixel_line_unchanged():
    # для середніх пікселів лінії A = 2, для кінців B = 1: нічого не видаляється
    img = np.full((5, 7), 255, dtype=np.uint8)
    img[2, 1:6] = 0
    assert np.array_equal(thin(img), img)


def test_thin_thick_line_becomes_thinner_but_not_empty():
    img = np.full((7, 11), 255, dtype=np.uint8)
    img[2:5, 1:10] = 0                      # лінія завтовшки 3
    out = thin(img)
    before, after = (img == 0), (out == 0)
    assert after.sum() > 0                  # лінія не зникла
    assert after.sum() < before.sum()       # стала тоншою
    assert not np.any(after & ~before)      # нових пікселів не з'явилось