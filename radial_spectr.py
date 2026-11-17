import glob
import numpy as np
from image_io import load_gray

for path in sorted(glob.glob("data/sample/*"))[:3]:
    img = load_gray(path).astype(float)
    img -= img.mean()
    power = np.abs(np.fft.fft2(img)) ** 2
    h, w = img.shape
    r = np.sqrt(np.fft.fftfreq(h)[:, None] ** 2 + np.fft.fftfreq(w)[None, :] ** 2)
    edges = np.linspace(0, 0.5, 26)
    print(path[-25:])
    for a, b in zip(edges[:-1], edges[1:]):
        sel = (r >= a) & (r < b)
        print(f"  {a:.2f}-{b:.2f}  середня потужність: {power[sel].mean():12.0f}")