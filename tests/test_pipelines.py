import numpy as np
from pipelines import run_pipeline


def _img():
    img = np.full((12, 12), 220, dtype=np.uint8)
    img[:, 4:7] = 40            # темна смуга
    return img


def test_pipeline1_returns_three_stages():
    out = run_pipeline("pipeline1", _img(), {})
    assert [name for name, _ in out] == ["stretch", "otsu", "thin"]
    assert out[-1][1].shape == (12, 12)


def test_pipeline2_saves_intermediate_files(tmp_path):
    params = {"fft": {"low": 0.18, "high": 0.45, "pad": 4},
              "sauvola": {"size": 5, "k": 0.1, "r": 128.0}}
    run_pipeline("pipeline2", _img(), params, save_dir=tmp_path, prefix="t")
    files = sorted(p.name for p in tmp_path.iterdir())
    assert files == ["t_s1_fft.png", "t_s2_sauvola.png", "t_s3_thin.png"]