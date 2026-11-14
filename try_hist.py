""" from image_io import load_gray, save_image
from methods.histogram import equalize_histogram
from methods.segmentation import otsu_threshold, otsu_binarize
from methods.morphology import erode, dilate, opening, closing

img = load_gray("data/1__M_Left_index_finger.BMP")
eq = equalize_histogram(img)
print("поріг без вирівнювання:", otsu_threshold(img))
print("поріг після вирівнювання:", otsu_threshold(eq))

save_image(otsu_binarize(img), "results/otsu_no_eq.png")
save_image(otsu_binarize(eq), "results/otsu_with_eq.png")

binary = otsu_binarize(equalize_histogram(img))

save_image(binary, "results/m0_binary.png")
save_image(erode(binary), "results/m1_erode.png")
save_image(dilate(binary), "results/m2_dilate.png")
save_image(opening(binary), "results/m3_opening.png")
save_image(closing(binary), "results/m4_closing.png")   

print("розмір:", binary.shape)
print("частка чорного:", (binary == 0).mean())
print("після закриття:", (closing(binary) == 0).mean())
from methods.histogram import equalize_histogram
eq = equalize_histogram(img)
for t in (90, 100, 110, 125):
    print(t, (eq <= t).mean())
    

import numpy as np
from methods.histogram import equalize_histogram
eq = equalize_histogram(img)

print("після вирівнювання:")
for t in (30, 50, 70, 80, 90):
    print(t, (eq <= t).mean())

print("без вирівнювання:")
for t in (60, 90, 110, 125, 140):
    print(t, (img <= t).mean())
    
save_image(np.where(eq <= 70, 0, 255), "results/eq_t70.png")
save_image(np.where(img <= 110, 0, 255), "results/raw_t110.png") """


import glob
from image_io import load_gray
from methods.histogram import equalize_histogram
from methods.segmentation import otsu_threshold

for path in sorted(glob.glob("data/sample/*")):
    img = load_gray(path)
    eq = equalize_histogram(img)
    raw_t, eq_t = otsu_threshold(img), otsu_threshold(eq)
    raw_f = (img <= raw_t).mean()
    eq_f = (eq <= eq_t).mean()
    print(f"{path[-30:]:30} без: {raw_f:.2f}  з: {eq_f:.2f}")