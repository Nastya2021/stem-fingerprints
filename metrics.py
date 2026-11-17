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

def _window_mean(arr, size):
    """Середнє у вікні size x size для кожного пікселя (краї: edge-padding)."""
    r = size // 2
    padded = np.pad(arr, r, mode="edge")
    h, w = arr.shape
    total = np.zeros((h, w))
    for dy in range(size):
        for dx in range(size):
            total += padded[dy:dy + h, dx:dx + w]
    return total / (size * size)


def ssim(a, b, size=7, max_value=255.0):
    """Індекс структурної подібності (SSIM) для двох зображень градацій сірого."""
    a = a.astype(np.float64)
    b = b.astype(np.float64)
    c1 = (0.01 * max_value) ** 2
    c2 = (0.03 * max_value) ** 2

    mx = _window_mean(a, size)
    my = _window_mean(b, size)
    var_x = np.maximum(_window_mean(a * a, size) - mx ** 2, 0)
    var_y = np.maximum(_window_mean(b * b, size) - my ** 2, 0)
    cov = _window_mean(a * b, size) - mx * my

    num = (2 * mx * my + c1) * (2 * cov + c2)
    den = (mx ** 2 + my ** 2 + c1) * (var_x + var_y + c2)
    return float(np.mean(num / den))