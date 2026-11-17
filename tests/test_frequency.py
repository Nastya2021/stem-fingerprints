import numpy as np
from methods.frequency import fft_bandpass, normalize_to_uint8


def test_bandpass_keeps_only_target_frequency():
    x = np.arange(16)
    row = 100 + 50 * np.cos(2 * np.pi * x / 4) + 30 * np.cos(2 * np.pi * x / 16)
    img = np.tile(row, (16, 1))
    expected = np.tile(50 * np.cos(2 * np.pi * x / 4), (16, 1))
    out = fft_bandpass(img, low=0.2, high=0.3)
    assert np.allclose(out, expected, atol=1e-9)


def test_bandpass_removes_constant_image():
    img = np.full((8, 8), 77, dtype=np.uint8)
    assert np.allclose(fft_bandpass(img, 0.06, 0.2), 0.0, atol=1e-9)


def test_normalize_range_and_constant():
    arr = np.array([[-1.0, 0.0], [1.0, 3.0]])
    out = normalize_to_uint8(arr)
    assert out.min() == 0 and out.max() == 255
    assert np.all(normalize_to_uint8(np.zeros((3, 3))) == 128)