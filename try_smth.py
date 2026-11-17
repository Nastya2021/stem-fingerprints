""" import glob
import numpy as np
from image_io import load_gray, save_image
from methods.segmentation import sauvola_binarize
from methods.morphology import thin

for i, path in enumerate(sorted(glob.glob("data/sample/*"))[:6]):
    img = load_gray(path)
    binary = sauvola_binarize(img, size=15, k=0.2)
    sk = thin(binary) == 0

    blocks = sk[:-1, :-1] & sk[1:, :-1] & sk[:-1, 1:] & sk[1:, 1:]
    print(i, "скелет:", sk.sum(), "| блоків 2x2:", int(blocks.sum()))

    # збільшення в 4 рази, щоб було видно окремі пікселі
    big = np.kron(np.where(sk, 0, 255), np.ones((4, 4), dtype=int))
    save_image(big, f"results/big{i}_skeleton.png")
    bigb = np.kron(binary, np.ones((4, 4), dtype=np.uint8))
    save_image(bigb, f"results/big{i}_binary.png") """
    
