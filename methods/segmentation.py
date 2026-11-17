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

def local_stats(image, size=15):
    """Локальне середнє і стандартне відхилення у вікні size x size."""
    r = size // 2
    img = image.astype(np.float64)
    padded = np.pad(img, r, mode="edge")   # край продовжуємо крайніми пікселями
    h, w = img.shape

    total = np.zeros((h, w))
    total_sq = np.zeros((h, w))
    for dy in range(size):
        for dx in range(size):
            window = padded[dy:dy + h, dx:dx + w]
            total += window
            total_sq += window ** 2

    n = size * size
    mean = total / n
    variance = np.maximum(total_sq / n - mean ** 2, 0)  # захист від -0.0000001
    return mean, np.sqrt(variance)


def sauvola_binarize(image, size=15, k=0.2, r=128.0):
    """Локальна бінаризація Саувола: гребені = 0, фон = 255."""
    mean, std = local_stats(image, size)
    threshold = mean * (1 + k * (std / r - 1))
    return np.where(image <= threshold, 0, 255).astype(np.uint8)