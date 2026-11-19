import numpy as np


def _to_mask(image):
    """Перетворює бінарне зображення на булеву маску: гребені (0) -> True, фон -> False."""
    return image == 0


def _from_mask(mask):
    """Перетворює булеву маску назад у зображення uint8: True -> 0 (гребінь), False -> 255 (фон)."""
    return np.where(mask, 0, 255).astype(np.uint8)


def _neighbors(mask, size, pad_value):
    """Повертає список зсунутих копій маски для вікна розміром size x size."""
    r = size // 2  # радіус вікна (для size=3 це 1 піксель навколо центрального)
    # Додаємо рамку завтовшки r навколо маски, заповнюючи її значенням pad_value
    padded = np.pad(mask, r, mode="constant", constant_values=pad_value)
    h, w = mask.shape
    shifted = []

    # Проходимо по всіх позиціях вікна size x size і беремо відповідний зсунутий зріз
    for dy in range(size):
        for dx in range(size):
            shifted.append(padded[dy:dy + h, dx:dx + w])
    return shifted


def erode(image, size=3):
    """Морфологічна ерозія: піксель лишається гребенем, лише якщо всі сусіди у вікні — гребені."""
    mask = _to_mask(image)
    # Поелементне логічне "І" по всіх зсунутих вікнах: True тільки там, де суцільний гребінь
    result = np.logical_and.reduce(_neighbors(mask, size, pad_value=False))
    return _from_mask(result)


def dilate(image, size=3):
    """Морфологічна дилатація: піксель стає гребенем, якщо хоч один сусід у вікні — гребінь."""
    mask = _to_mask(image)
    # Поелементне логічне "АБО" по всіх зсунутих вікнах: True, якщо поруч є хоча б один гребінь
    result = np.logical_or.reduce(_neighbors(mask, size, pad_value=False))
    return _from_mask(result)


def opening(image, size=3):
    """Морфологічне відкриття: спочатку ерозія, потім дилатація (прибирає дрібні шумові крапки)."""
    return dilate(erode(image, size), size)


def closing(image, size=3):
    """Морфологічне закриття: спочатку дилатація, потім ерозія (заповнює дрібні розриви в лініях)."""
    return erode(dilate(image, size), size)


def _thin_step(mask, step):
    """Виконує один підкрок алгоритму стоншення Жанга-Суена та повертає оновлену маску."""
    # Додаємо рамку з False в 1 піксель, щоб коректно брати 8 сусідів навіть біля країв
    p = np.pad(mask, 1, mode="constant", constant_values=False)
    h, w = mask.shape

    # Виділяємо 8 сусідів навколо кожного пікселя за годинниковою стрілкою (починаючи зі зсуву вгору)
    p2 = p[0:h, 1:w + 1]      # верхній сусід (N)
    p3 = p[0:h, 2:w + 2]      # верхній правий (NE)
    p4 = p[1:h + 1, 2:w + 2]  # правий (E)
    p5 = p[2:h + 2, 2:w + 2]  # нижній правий (SE)
    p6 = p[2:h + 2, 1:w + 1]  # нижній (S)
    p7 = p[2:h + 2, 0:w]      # нижній лівий (SW)
    p8 = p[1:h + 1, 0:w]      # лівий (W)
    p9 = p[0:h, 0:w]          # верхній лівий (NW)

    ring = [p2, p3, p4, p5, p6, p7, p8, p9]
    # B: кількість сусідів серед 8 навколишніх, які є пікселями гребеня (True)
    b = sum(n.astype(int) for n in ring)

    # Замикаємо кільце сусідів (додаємо p2 у кінець), щоб перевірити перехід p9 -> p2
    ring_closed = ring + [p2]
    # A: кількість переходів "не гребінь (False) -> гребінь (True)" при обході кільця сусідів
    a = sum((~ring_closed[i] & ring_closed[i + 1]).astype(int) for i in range(8))

    # Додаткові умови для збереження зв'язності залежно від номера підкроку (0 або 1)
    if step == 0:
        cond = ~(p2 & p4 & p6) & ~(p4 & p6 & p8)
    else:
        cond = ~(p2 & p4 & p8) & ~(p2 & p6 & p8)

    # Піксель видаляється, якщо він є гребенем і задовольняє всі критерії Жанга-Суена
    remove = mask & (b >= 2) & (b <= 6) & (a == 1) & cond
    # Повертаємо маску без видалених пікселів
    return mask & ~remove


def thin(image):
    """Ітеративне стоншення Жанга-Суена: зводить гребені (0) до ліній завтовшки ~1 піксель."""
    mask = _to_mask(image)
    while True:
        before = mask.copy()       # зберігаємо стан маски до початку повної ітерації
        mask = _thin_step(mask, 0) # перший підкрок
        mask = _thin_step(mask, 1) # другий підкрок
        # Якщо маска більше не змінюється — стоншення завершене
        if np.array_equal(mask, before):
            break
    return _from_mask(mask)