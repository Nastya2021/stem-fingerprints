import numpy as np
from degrade import gaussian_noise, salt_pepper, brightness_gradient, reduce_contrast


def test_gaussian_is_reproducible():
    img = np.full((8, 8), 100, dtype=np.uint8)
    assert np.array_equal(gaussian_noise(img, 20, seed=1), gaussian_noise(img, 20, seed=1))


def test_gaussian_zero_sigma_is_identity():
    img = np.array([[10, 200], [30, 40]], dtype=np.uint8)
    assert np.array_equal(gaussian_noise(img, 0), img)


def test_salt_pepper_only_extremes_changed():
    img = np.full((50, 50), 100, dtype=np.uint8)
    out = salt_pepper(img, 0.1, seed=3)
    changed = out != img
    assert set(np.unique(out[changed])) <= {0, 255}
    assert 0.05 < changed.mean() < 0.15


def test_gradient_hand_calc():
    img = np.full((1, 3), 100, dtype=np.uint8)
    # амплітуда 40: зсуви -20, 0, +20
    assert np.array_equal(brightness_gradient(img, 40), np.array([[80, 100, 120]], dtype=np.uint8))


def test_reduce_contrast_hand_calc():
    img = np.array([[0, 100], [200, 100]], dtype=np.uint8)   # середнє 100
    out = reduce_contrast(img, 0.5)
    assert np.array_equal(out, np.array([[50, 100], [150, 100]], dtype=np.uint8))