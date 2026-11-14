import numpy as np


def otsu_threshold(image):
    """Повертає поріг Оцу для зображення uint8 (градації сірого)."""
    hist = np.bincount(image.ravel(), minlength=256)
    prob = hist / image.size            # частка пікселів кожної яскравості
    levels = np.arange(256)

    w0 = np.cumsum(prob)                # частка класу 0 для кожного порога
    mean_cum = np.cumsum(prob * levels) # накопичена "сума яскравостей"
    mean_total = mean_cum[-1]           # середня яскравість усього зображення

    w1 = 1.0 - w0
    # ділимо тільки там, де обидва класи непорожні
    valid = (w0 > 0) & (w1 > 0)
    sigma = np.zeros(256)
    m0 = mean_cum[valid] / w0[valid]
    m1 = (mean_total - mean_cum[valid]) / w1[valid]
    sigma[valid] = w0[valid] * w1[valid] * (m0 - m1) ** 2

    return int(np.argmax(sigma))


def otsu_binarize(image):
    """Бінаризація за Оцу: гребені (темні) = 0, фон = 255."""
    t = otsu_threshold(image)
    return np.where(image <= t, 0, 255).astype(np.uint8)