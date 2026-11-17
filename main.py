import argparse
import glob
import json
import os

from image_io import load_gray
from pipelines import PIPELINES, run_pipeline

IMAGE_EXTENSIONS = (".bmp", ".png", ".jpg", ".jpeg", ".tif", ".tiff")


def load_config(path):
    """Читає конфігураційний файл JSON."""
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def list_images(folder):
    """Повертає відсортований список файлів-зображень у папці."""
    files = [
        p for p in glob.glob(os.path.join(folder, "*"))
        if p.lower().endswith(IMAGE_EXTENSIONS)
    ]
    return sorted(files)


def process_folder(pipeline_name, config):
    """Запускає пайплайн на всіх зображеннях із input_dir."""
    params = config["pipelines"][pipeline_name]
    input_dir = config["input_dir"]
    output_dir = os.path.join(config["output_dir"], pipeline_name)

    paths = list_images(input_dir)
    if not paths:
        print(f"У папці {input_dir} немає зображень.")
        return 0

    done = 0
    for path in paths:
        name = os.path.splitext(os.path.basename(path))[0]
        try:
            image = load_gray(path)
            run_pipeline(pipeline_name, image, params,
                         save_dir=os.path.join(output_dir, name), prefix=name)
            done += 1
        except Exception as error:   # пошкоджений файл не зупиняє решту
            print(f"Пропущено {path}: {error}")
    print(f"{pipeline_name}: оброблено {done} із {len(paths)}")
    return done


def main():
    parser = argparse.ArgumentParser(description="Обробка відбитків пальців")
    parser.add_argument("--pipeline", choices=list(PIPELINES) + ["all"],
                        default="all", help="який пайплайн запустити")
    parser.add_argument("--config", default="config.json",
                        help="шлях до конфігураційного файлу")
    args = parser.parse_args()

    config = load_config(args.config)
    names = list(PIPELINES) if args.pipeline == "all" else [args.pipeline]
    for name in names:
        process_folder(name, config)


if __name__ == "__main__":
    main()