import glob
from image_io import load_gray, save_image
from methods.segmentation import otsu_binarize, sauvola_binarize

for i, path in enumerate(sorted(glob.glob("data/sample/*"))[:6]):
    img = load_gray(path)
    otsu = otsu_binarize(img)
    local = sauvola_binarize(img, size=15, k=0.2)
    print(i, "Оцу чорного:", round((otsu == 0).mean(), 2),
             "Саувола чорного:", round((local == 0).mean(), 2))
    save_image(otsu, f"results/cmp{i}_otsu.png")
    save_image(local, f"results/cmp{i}_sauvola.png")