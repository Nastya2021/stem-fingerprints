import argparse
import csv
import os

from degrade import degrade
from image_io import load_gray
from main import list_images, load_config
from metrics import mse, psnr, ssim
from pipelines import PIPELINES, run_pipeline

FIELDS = ["image", "pipeline", "degradation", "level", "stage", "mse", "psnr", "ssim"]


def run_experiment(config, pipeline_names):
    """Повертає список рядків з метриками для всіх комбінацій."""
    deg = config["degradations"]
    seed = deg.get("seed", 0)
    kinds = [k for k in deg if k != "seed"]
    rows = []

    for path in list_images(config["input_dir"]):
        name = os.path.splitext(os.path.basename(path))[0]
        try:
            clean = load_gray(path)
        except Exception as error:
            print(f"Пропущено {path}: {error}")
            continue

        for pname in pipeline_names:
            params = config["pipelines"][pname]
            ref = run_pipeline(pname, clean, params)
            # стадії: передостання = бінарна, остання = скелет
            ref_stages = {"binary": ref[-2][1], "skeleton": ref[-1][1]}

            for kind in kinds:
                for level in deg[kind]:
                    bad = degrade(clean, kind, level, seed)
                    out = run_pipeline(pname, bad, params)
                    out_stages = {"binary": out[-2][1], "skeleton": out[-1][1]}
                    for stage in ("binary", "skeleton"):
                        a, b = ref_stages[stage], out_stages[stage]
                        rows.append({
                            "image": name, "pipeline": pname,
                            "degradation": kind, "level": level, "stage": stage,
                            "mse": round(mse(a, b), 4),
                            "psnr": round(psnr(a, b), 4),
                            "ssim": round(ssim(a, b), 4),
                        })
        print(f"оброблено {name}")
    return rows


def save_csv(rows, path):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description="Експеримент зі стійкості пайплайнів")
    parser.add_argument("--config", default="config.json")
    parser.add_argument("--pipelines", nargs="+", default=None,
                        help="які пайплайни порівнювати (за замовчуванням усі)")
    args = parser.parse_args()

    config = load_config(args.config)
    names = args.pipelines or list(PIPELINES)
    rows = run_experiment(config, names)
    out_path = os.path.join(config["output_dir"], "metrics.csv")
    save_csv(rows, out_path)
    print(f"Збережено {len(rows)} рядків у {out_path}")


if __name__ == "__main__":
    main()