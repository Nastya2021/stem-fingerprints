import numpy as np
from image_io import load_gray, save_image


def test_save_and_load_roundtrip(tmp_path):
    original = np.array([[0, 100], [200, 255]], dtype=np.uint8)
    path = tmp_path / "test.png"
    save_image(original, path)
    loaded = load_gray(path)
    assert loaded.shape == (2, 2)
    assert np.array_equal(loaded, original)


def test_save_clips_values(tmp_path):
    array = np.array([[-10, 300]])
    path = tmp_path / "clip.png"
    save_image(array, path)
    loaded = load_gray(path)
    assert loaded[0, 0] == 0
    assert loaded[0, 1] == 255