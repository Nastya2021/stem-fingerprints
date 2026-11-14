import numpy as np


def _to_mask(image):
    """Гребені (значення 0) -> True, фон -> False."""
    return image == 0


def _from_mask(mask):
    """True -> 0 (чорний гребінь), False -> 255 (білий фон)."""
    return np.where(mask, 0, 255).astype(np.uint8)


def _neighbors(mask, size, pad_value):
    """Повертає список зсунутих копій маски для вікна size x size."""
    r = size // 2
    padded = np.pad(mask, r, mode="constant", constant_values=pad_value)
    h, w = mask.shape
    shifted = []
    for dy in range(size):
        for dx in range(size):
            shifted.append(padded[dy:dy + h, dx:dx + w])
    return shifted


def erode(image, size=3):
    """Ерозія: піксель лишається гребенем, якщо всі сусіди у вікні гребені."""
    mask = _to_mask(image)
    result = np.logical_and.reduce(_neighbors(mask, size, pad_value=False))
    return _from_mask(result)


def dilate(image, size=3):
    """Дилатація: піксель стає гребенем, якщо хоч один сусід у вікні гребінь."""
    mask = _to_mask(image)
    result = np.logical_or.reduce(_neighbors(mask, size, pad_value=False))
    return _from_mask(result)


def opening(image, size=3):
    """Відкриття: ерозія, потім дилатація (прибирає дрібні крапки)."""
    return dilate(erode(image, size), size)


def closing(image, size=3):
    """Закриття: дилатація, потім ерозія (заповнює дрібні розриви)."""
    return erode(dilate(image, size), size)