import numpy as np


def mse(a, b):
    """Середньоквадратична помилка між двома зображеннями."""
    a = a.astype(np.float64)
    b = b.astype(np.float64)
    return np.mean((a - b) ** 2)


def psnr(a, b, max_value=255.0):
    """Пікове відношення сигнал/шум у децибелах."""
    error = mse(a, b)
    if error == 0:
        return float("inf")
    return 10 * np.log10(max_value ** 2 / error)