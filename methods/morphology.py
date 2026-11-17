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

def _thin_step(mask, step):
    """Один підкрок Zhang-Suen. Повертає нову маску."""
    p = np.pad(mask, 1, mode="constant", constant_values=False)
    h, w = mask.shape
    p2 = p[0:h, 1:w + 1]
    p3 = p[0:h, 2:w + 2]
    p4 = p[1:h + 1, 2:w + 2]
    p5 = p[2:h + 2, 2:w + 2]
    p6 = p[2:h + 2, 1:w + 1]
    p7 = p[2:h + 2, 0:w]
    p8 = p[1:h + 1, 0:w]
    p9 = p[0:h, 0:w]

    ring = [p2, p3, p4, p5, p6, p7, p8, p9]
    b = sum(n.astype(int) for n in ring)

    ring_closed = ring + [p2]
    a = sum((~ring_closed[i] & ring_closed[i + 1]).astype(int) for i in range(8))

    if step == 0:
        cond = ~(p2 & p4 & p6) & ~(p4 & p6 & p8)
    else:
        cond = ~(p2 & p4 & p8) & ~(p2 & p6 & p8)

    remove = mask & (b >= 2) & (b <= 6) & (a == 1) & cond
    return mask & ~remove


def thin(image):
    """Стоншення Zhang-Suen: гребені (0) -> лінії завтовшки 1 піксель."""
    mask = _to_mask(image)
    while True:
        before = mask.copy()
        mask = _thin_step(mask, 0)
        mask = _thin_step(mask, 1)
        if np.array_equal(mask, before):
            break
    return _from_mask(mask)