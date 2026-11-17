import os

from image_io import save_image
from methods.histogram import equalize_histogram, stretch_contrast
from methods.segmentation import otsu_binarize, sauvola_binarize
from methods.morphology import thin
from methods.frequency import fft_enhance

# Реєстр стадій: назва -> функція (image, **params) -> image
STAGES = {
    "equalize": lambda img, **p: equalize_histogram(img),
    "otsu": lambda img, **p: otsu_binarize(img),
    "thin": lambda img, **p: thin(img),
    "fft": lambda img, **p: fft_enhance(img, **p),
    "sauvola": lambda img, **p: sauvola_binarize(img, **p),
    "stretch": lambda img, **p: stretch_contrast(img, **p)
}

# Порядок стадій кожного пайплайна
PIPELINES = {
    "pipeline1": ["stretch", "otsu", "thin"],
    "pipeline1_eq": ["equalize", "otsu", "thin"],
    "pipeline2": ["fft", "sauvola", "thin"],
}


def run_pipeline(name, image, params, save_dir=None, prefix="img"):
    """Проганяє зображення через усі стадії пайплайна.

    params: словник параметрів стадій (з конфіга).
    save_dir: якщо задано, зберігає результат кожної стадії.
    Повертає список (назва_стадії, результат) для всіх стадій.
    """
    results = []
    current = image
    for i, stage in enumerate(PIPELINES[name], start=1):
        current = STAGES[stage](current, **params.get(stage, {}))
        results.append((stage, current))
        if save_dir is not None:
            os.makedirs(save_dir, exist_ok=True)
            save_image(current, os.path.join(save_dir, f"{prefix}_s{i}_{stage}.png"))
    return results