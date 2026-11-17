import numpy as np
from metrics import mse, psnr


def test_mse_simple():
    a = np.array([[0, 0], [0, 0]], dtype=np.uint8)
    b = np.array([[2, 2], [2, 2]], dtype=np.uint8)
    # кожна різниця 2, квадрат 4, середнє 4
    assert mse(a, b) == 4.0


def test_mse_no_overflow():
    a = np.array([[50]], dtype=np.uint8)
    b = np.array([[200]], dtype=np.uint8)
    assert mse(a, b) == 150 ** 2


def test_psnr_identical_is_inf():
    a = np.array([[10, 20], [30, 40]], dtype=np.uint8)
    assert psnr(a, a) == float("inf")
    
from metrics import ssim

def test_ssim_identical_is_one():
    a = np.array([[10, 200, 30], [40, 50, 60], [70, 80, 250]], dtype=np.uint8)
    assert np.isclose(ssim(a, a, size=3), 1.0)

def test_ssim_constant_images_hand_calc():
    # дві константи 100 і 200: var = cov = 0
    # SSIM = (2*100*200 + C1) / (100^2 + 200^2 + C1), C1 = 6.5025
    a = np.full((5, 5), 100, dtype=np.uint8)
    b = np.full((5, 5), 200, dtype=np.uint8)
    c1 = (0.01 * 255) ** 2
    expected = (2 * 100 * 200 + c1) / (100 ** 2 + 200 ** 2 + c1)
    assert np.isclose(ssim(a, b, size=3), expected)

def test_ssim_inverted_is_lower_than_identical():
    a = np.tile(np.array([0, 255], dtype=np.uint8), (6, 3))
    assert ssim(a, 255 - a, size=3) < 0.1