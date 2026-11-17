import glob
import numpy as np
from image_io import load_gray
from methods.segmentation import sauvola_binarize
from methods.morphology import thin

for i, path in enumerate(sorted(glob.glob("data/sample/*"))[:6]):
    img = load_gray(path)
    binary = sauvola_binarize(img, size=15, k=0.2)
    sk = thin(binary) == 0

    p = np.pad(sk, 1)
    h, w = sk.shape
    neighbors = sum(
        p[dy:dy + h, dx:dx + w].astype(int)
        for dy in range(3) for dx in range(3) if (dy, dx) != (1, 1)
    )
    n = sk.sum()
    print(i,
          "скелет:", n,
          "| ізольовані:", int((sk & (neighbors == 0)).sum()),
          "| кінці (1 сусід):", int((sk & (neighbors == 1)).sum()),
          "| 3+ сусіди:", int((sk & (neighbors >= 3)).sum()))