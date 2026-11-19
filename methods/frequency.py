import numpy as np


def fft_bandpass(image, low=0.06, high=0.2, pad=16):
    """Смуговий частотний фільтр: пропускає частоти в межах [low, high] (у циклах на піксель).

    Повертає масив float (нульова частота — середня яскравість — відсікається).
    """
    # Додаємо дзеркально відбиту рамку завтовшки pad навколо зображення, щоб краї не давали артефактів
    img = np.pad(image.astype(np.float64), pad, mode="reflect")

    h, w = image.shape
    # Переводимо зображення у частотну область за допомогою 2D швидкого перетворення Фур'є
    # (Примітка: тут spectrum рахується від початкового image, а не від img з паддінгом)
    spectrum = np.fft.fft2(image.astype(np.float64))

    # Масиви просторових частот уздовж вертикальної (fy) та горизонтальної (fx) осей
    fy = np.fft.fftfreq(h)[:, None]  # стовпець розміру (h, 1)
    fx = np.fft.fftfreq(w)[None, :]  # рядок розміру (1, w)

    # Радіальна відстань (частота) від центру частотної площини для кожної точки
    radius = np.sqrt(fx ** 2 + fy ** 2)

    # Маска-кільце: True лише для частот між low і high (інші частоти зануляться)
    mask = (radius >= low) & (radius <= high)

    # Множимо спектр на маску і повертаємося назад у просторову область (беремо дійсну частину)
    return np.real(np.fft.ifft2(spectrum * mask))


def normalize_to_uint8(array):
    """Лінійно масштабує значення масиву у діапазон 0..255.
    Якщо всі значення однакові (константний масив), повертає масив із 128."""
    lo, hi = array.min(), array.max()
    # Захист від ділення на нуль, якщо в масиві немає перепадів яскравості
    if hi - lo < 1e-12:
        return np.full(array.shape, 128, dtype=np.uint8)

    # Нормуємо в [0, 1], множимо на 255, округлюємо та приводимо до типу uint8
    return np.round((array - lo) / (hi - lo) * 255).astype(np.uint8)


def fft_enhance(image, low=0.06, high=0.2, pad=16):
    """Повний цикл покращення: смугова FFT-фільтрація + нормування до uint8 (0..255)."""
    return normalize_to_uint8(fft_bandpass(image, low, high))