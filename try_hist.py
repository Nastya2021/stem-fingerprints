from image_io import load_gray, save_image
from methods.histogram import equalize_histogram
from methods.segmentation import otsu_threshold, otsu_binarize

img = load_gray("data/1__M_Left_index_finger.BMP")
eq = equalize_histogram(img)
print("поріг без вирівнювання:", otsu_threshold(img))
print("поріг після вирівнювання:", otsu_threshold(eq))

save_image(otsu_binarize(img), "results/otsu_no_eq.png")
save_image(otsu_binarize(eq), "results/otsu_with_eq.png")

