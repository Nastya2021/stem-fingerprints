import argparse
import os

import matplotlib
matplotlib.use("Agg")           # малюємо у файли, без вікон
import matplotlib.pyplot as plt
import pandas as pd

PSNR_CAP = 100.0                # inf (ідентичні картинки) замінюємо на 100


def load_metrics(path):
    df = pd.read_csv(path)
    df["psnr"] = df["psnr"].replace(float("inf"), PSNR_CAP)
    return df


def summarize(df):
    """Середнє й розкид метрик по відбитках для кожної комбінації."""
    keys = ["pipeline", "degradation", "level", "stage"]
    out = df.groupby(keys)[["mse", "psnr", "ssim"]].agg(["mean", "std"])
    out.columns = [f"{m}_{s}" for m, s in out.columns]
    return out.reset_index()


def best_pipeline(summary, metric="ssim"):
    """Для кожної умови: який пайплайн має найкраще середнє значення метрики."""
    keys = ["degradation", "level", "stage"]
    col = f"{metric}_mean"
    idx = summary.groupby(keys)[col].idxmax()
    return summary.loc[idx, keys + ["pipeline", col]].reset_index(drop=True)


import numpy as np

PIPE_NAMES = {
    "pipeline1": "П1: розтягування + Оцу",
    "pipeline1_eq": "П1-eq: вирівнювання + Оцу",
    "pipeline2": "П2: FFT + Саувола",
}
DEG_NAMES = {
    "gaussian": "Гаусів шум",
    "salt_pepper": "Сіль і перець",
    "gradient": "Градієнт яскравості",
    "contrast": "Низький контраст",
}
DEG_XLABEL = {
    "gaussian": "σ шуму (більше = сильніше псування)",
    "salt_pepper": "частка зіпсованих пікселів (більше = сильніше)",
    "gradient": "амплітуда градієнта (більше = сильніше)",
    "contrast": "коефіцієнт контрасту (менше = сильніше псування)",
}


def pick_middle_level(summary):
    """Лишає для кожного виду псування лише середній рівень."""
    parts = []
    for _, g in summary.groupby("degradation"):
        levels = sorted(g["level"].unique())
        parts.append(g[g["level"] == levels[len(levels) // 2]])
    return pd.concat(parts)


def plot_bars(summary, out_dir, stage="skeleton"):
    """Стовпчики: SSIM для кожного виду псування і кожного пайплайна."""
    os.makedirs(out_dir, exist_ok=True)
    part = pick_middle_level(summary[summary["stage"] == stage])
    means = part.pivot_table(index="degradation", columns="pipeline", values="ssim_mean")
    stds = part.pivot_table(index="degradation", columns="pipeline", values="ssim_std")

    kinds, pipes = list(means.index), list(means.columns)
    width = 0.8 / len(pipes)
    x = np.arange(len(kinds))

    fig, ax = plt.subplots(figsize=(8, 4.8))
    for j, p in enumerate(pipes):
        m, s = means[p].values, stds[p].fillna(0).values
        err = [np.minimum(s, m), np.minimum(s, 1 - m)]   # не виходимо за 0..1
        pos = x + (j - (len(pipes) - 1) / 2) * width
        bars = ax.bar(pos, m, width, yerr=err, capsize=3, label=PIPE_NAMES.get(p, p))
        for rect, val in zip(bars, m):
            ax.text(rect.get_x() + rect.get_width() / 2, 0.03, f"{val:.2f}",
                    ha="center", va="bottom", color="white", fontsize=9, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels([DEG_NAMES.get(k, k) for k in kinds])
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("SSIM (1 = псування не зіпсувало результат)")
    ax.set_title("Стійкість пайплайнів до псування (скелет, середній рівень)")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.1), ncol=3)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "bars_ssim_skeleton.png"), dpi=150)
    plt.close(fig)


def plot_lines(summary, out_dir, stage="skeleton"):
    """Лінії: як SSIM падає зі зростанням псування (лише якщо рівнів ≥ 2)."""
    os.makedirs(out_dir, exist_ok=True)
    for kind, g in summary[summary["stage"] == stage].groupby("degradation"):
        if g["level"].nunique() < 2:
            continue
        fig, ax = plt.subplots(figsize=(5.5, 4))
        for pname, p in g.groupby("pipeline"):
            p = p.sort_values("level")
            m, s = p["ssim_mean"].values, p["ssim_std"].fillna(0).values
            line, = ax.plot(p["level"], m, marker="o", label=PIPE_NAMES.get(pname, pname))
            ax.fill_between(p["level"], np.clip(m - s, 0, 1), np.clip(m + s, 0, 1),
                            color=line.get_color(), alpha=0.12)
        if kind == "contrast":
            ax.invert_xaxis()
        ax.set_ylim(0, 1.05)
        ax.set_xlabel(DEG_XLABEL.get(kind, kind))
        ax.set_ylabel("SSIM")
        ax.set_title(f"{DEG_NAMES.get(kind, kind)} (скелет)")
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, f"lines_ssim_{kind}.png"), dpi=150)
        plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Аналіз metrics.csv")
    parser.add_argument("--csv", default="results/metrics.csv")
    parser.add_argument("--out", default="results/analysis")
    args = parser.parse_args()

    df = load_metrics(args.csv)
    summary = summarize(df)
    os.makedirs(args.out, exist_ok=True)
    summary.to_csv(os.path.join(args.out, "summary.csv"), index=False)
    best_pipeline(summary).to_csv(os.path.join(args.out, "best.csv"), index=False)
    plot_bars(summary, args.out)
    plot_lines(summary, args.out)

def compact_table(summary, stage="skeleton", metric="ssim"):
    part = pick_middle_level(summary[summary["stage"] == stage])
    table = part.pivot_table(index="degradation", columns="pipeline",
                             values=f"{metric}_mean")
    return table.round(2)

if __name__ == "__main__":
    main()
    