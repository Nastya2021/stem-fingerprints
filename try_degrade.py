from image_io import load_gray, save_image
from degrade import degrade

img = load_gray("data/sample/1__M_Left_little_finger.BMP")
for kind, level in [("gaussian", 25), ("salt_pepper", 0.05), ("gradient", 80), ("contrast", 0.5)]:
    save_image(degrade(img, kind, level, seed=42), f"results/deg_{kind}.png")