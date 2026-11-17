import glob
from image_io import load_gray, save_image
from methods.frequency import fft_enhance
from methods.segmentation import sauvola_binarize
from methods.morphology import thin

for i, path in enumerate(sorted(glob.glob("data/sample/*"))[:6]):
    img = load_gray(path)
    enh = fft_enhance(img, low=0.18, high=0.45)   # підстав свої значенняlow=0.18, high=0.45, k=0.1
    save_image(enh, "results/edge_enh.png")
    save_image(sauvola_binarize(enh, size=15, k=0.1), "results/edge_binary.png")
    binary = sauvola_binarize(enh, size=15, k=0.1)
    save_image(enh, f"results/fft{i}_enh.png")
    save_image(binary, f"results/fft{i}_binary.png")
    save_image(thin(binary), f"results/fft{i}_skeleton.png")
    print(i, "чорного:", round((binary == 0).mean(), 2)) 