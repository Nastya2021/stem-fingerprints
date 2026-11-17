import numpy as np


def equalize_histogram(image):
    """Гістограмне вирівнювання для зображення uint8 (градації сірого).
    Гістограма: порахувати, скільки пікселів має кожну яскравість від 0 до 255.
    CDF (кумулятивна функція розподілу): для кожної яскравості скласти кількість пікселів з яскравістю не більшою за неї.
    Нормування: перетворити CDF так, щоб найменша наявна яскравість стала 0, а найбільша 255.
    Таблиця відповідності: замінити кожен піксель на нове значення з таблиці."""
    # 1. Гістограма: hist[v] = кількість пікселів з яскравістю v
    hist = np.bincount(image.ravel(), minlength=256)

    # 2. Кумулятивна сума
    cdf = np.cumsum(hist)

    # 3. Нормування
    n = image.size
    cdf_min = cdf[cdf > 0].min()
    if n == cdf_min:
        # усі пікселі однакові: вирівнювати нічого
        return image.copy()
    table = np.round((cdf - cdf_min) / (n - cdf_min) * 255)
    table = np.clip(table, 0, 255).astype(np.uint8)

    # 4. Заміна кожного пікселя за таблицею
    return table[image]

def stretch_contrast(image, low_pct=1, high_pct=99):
    """Лінійне розтягування контрасту: low_pct і high_pct перцентилі -> 0 і 255."""
    lo, hi = np.percentile(image, [low_pct, high_pct])
    if hi - lo < 1e-9:
        return image.copy()
    out = (image.astype(np.float64) - lo) / (hi - lo) * 255
    return np.clip(np.round(out), 0, 255).astype(np.uint8)