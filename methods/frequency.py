import numpy as np


def fft_bandpass(image, low=0.06, high=0.2, pad=16):
    """Смуговий фільтр: лишає частоти від low до high (цикли на піксель).

    Повертає масив float (середнє значення прибрано, бо частота 0 відсікається).
    """
    img = np.pad(image.astype(np.float64), pad, mode="reflect")

    h, w = image.shape
    spectrum = np.fft.fft2(image.astype(np.float64))

    fy = np.fft.fftfreq(h)[:, None]     # частоти по вертикалі
    fx = np.fft.fftfreq(w)[None, :]     # частоти по горизонталі
    radius = np.sqrt(fx ** 2 + fy ** 2)

    mask = (radius >= low) & (radius <= high)
    return np.real(np.fft.ifft2(spectrum * mask))


def normalize_to_uint8(array):
    """Розтягує значення до 0-255. Для константного масиву повертає 128."""
    lo, hi = array.min(), array.max()
    if hi - lo < 1e-12:
        return np.full(array.shape, 128, dtype=np.uint8)
    return np.round((array - lo) / (hi - lo) * 255).astype(np.uint8)


def fft_enhance(image, low=0.06, high=0.2, pad=16):
    """Фільтр + нормування: готова uint8-картинка для наступної стадії."""
    return normalize_to_uint8(fft_bandpass(image, low, high))