import numpy as np
from PIL import Image


def load_gray(path):
    """Читає зображення і повертає двовимірний масив uint8 (0-255)."""
    img = Image.open(path).convert("L")  # "L" = градації сірого
    return np.array(img, dtype=np.uint8)


def save_image(array, path):
    """Зберігає двовимірний масив як зображення.

    Значення обрізаються до діапазону 0-255 і перетворюються в uint8.
    """
    array = np.clip(array, 0, 255).astype(np.uint8)
    Image.fromarray(array).save(path)