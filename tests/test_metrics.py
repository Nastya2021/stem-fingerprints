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