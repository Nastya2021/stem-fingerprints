import glob
from image_io import load_gray, save_image
from methods.segmentation import sauvola_binarize
from methods.morphology import thin

for i, path in enumerate(sorted(glob.glob("data/sample/*"))[:6]):
    img = load_gray(path)
    binary = sauvola_binarize(img, size=15, k=0.2)
    skeleton = thin(binary)
    print(i, "чорного до:", (binary == 0).sum(), "після:", (skeleton == 0).sum())
    save_image(binary, f"results/th{i}_binary.png")
    save_image(skeleton, f"results/th{i}_skeleton.png")