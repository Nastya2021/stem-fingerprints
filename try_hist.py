from image_io import load_gray, save_image
from methods.histogram import equalize_histogram

img = load_gray("data/1__M_Left_index_finger.BMP")
save_image(equalize_histogram(img), "results/hist_test.png")