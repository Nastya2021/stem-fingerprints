import numpy as np


def _clip(arr):
    return np.clip(np.round(arr), 0, 255).astype(np.uint8)


def gaussian_noise(image, sigma, seed=0):
    """Додає гаусів шум з відхиленням sigma."""
    rng = np.random.default_rng(seed)
    noise = rng.normal(0.0, sigma, image.shape)
    return _clip(image.astype(np.float64) + noise)


def salt_pepper(image, amount, seed=0):
    """Частка amount пікселів стає 0 або 255 (порівну)."""
    rng = np.random.default_rng(seed)
    out = image.copy()
    r = rng.random(image.shape)
    out[r < amount / 2] = 0
    out[(r >= amount / 2) & (r < amount)] = 255
    return out


def brightness_gradient(image, amplitude):
    """Додає лінійний градієнт зліва направо від -amplitude/2 до +amplitude/2."""
    w = image.shape[1]
    ramp = np.linspace(-amplitude / 2, amplitude / 2, w)[None, :]
    return _clip(image.astype(np.float64) + ramp)


def reduce_contrast(image, factor):
    """Стискає контраст до середнього: mean + factor * (img - mean)."""
    img = image.astype(np.float64)
    mean = img.mean()
    return _clip(mean + factor * (img - mean))


DEGRADATIONS = {
    "gaussian": lambda img, level, seed: gaussian_noise(img, level, seed),
    "salt_pepper": lambda img, level, seed: salt_pepper(img, level, seed),
    "gradient": lambda img, level, seed: brightness_gradient(img, level),
    "contrast": lambda img, level, seed: reduce_contrast(img, level),
}


def degrade(image, kind, level, seed=0):
    """Застосовує псування kind із рівнем level."""
    return DEGRADATIONS[kind](image, level, seed)