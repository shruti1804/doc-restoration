"""Run: python evaluate.py  -> ablation table (each stage's contribution) on synthetic data.

For every severity ('light', 'medium', 'heavy') and 10 random seeds, every variant is
run on a degraded page and compared with the clean page. The averages are saved to
outputs\\results.csv and a bar chart to outputs\\ablation.png.
"""
import csv
import os

import numpy as np

from degrade import make_clean_page, degrade
from restore import steps
from restore.metrics import psnr_ssim
from restore.pipeline import restore

SEVERITIES = ["light", "medium", "heavy"]
SEEDS = list(range(10))
OUT_DIR = "outputs"

# Each variant is a function: degraded image -> processed image.
VARIANTS = {
    "degraded (no processing)": lambda bad: bad,
    "CLAHE only": lambda bad: steps.enhance_contrast(bad),
    "Background correction only": lambda bad: steps.correct_background(bad),
    "Background + denoise": lambda bad: steps.denoise(steps.correct_background(bad)),
    "Full pipeline (smart routing)": lambda bad: restore(bad, force_all=False)[0],
    "Full pipeline (all steps forced)": lambda bad: restore(bad, force_all=True)[0],
}


def run_evaluation():
    """Return a list of result rows, one per (severity, method), averaged over the seeds."""
    clean = make_clean_page()
    rows = []
    for severity in SEVERITIES:
        scores = {name: [] for name in VARIANTS}
        for seed in SEEDS:
            bad = degrade(clean, seed, severity)
            for name, fn in VARIANTS.items():
                m = psnr_ssim(fn(bad), clean)
                scores[name].append((m["PSNR_dB"], m["SSIM"]))
        for name, vals in scores.items():
            arr = np.array(vals)
            rows.append({
                "severity": severity,
                "method": name,
                "psnr_db": round(float(arr[:, 0].mean()), 2),
                "ssim": round(float(arr[:, 1].mean()), 4),
                "psnr_std": round(float(arr[:, 0].std()), 2),
                "ssim_std": round(float(arr[:, 1].std()), 4),
                "n_seeds": len(SEEDS),
            })
        print(f"finished severity '{severity}' ({len(SEEDS)} seeds)")
    return rows


def save_csv(rows, path):
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def print_table(rows):
    for severity in SEVERITIES:
        print(f"\nSeverity: {severity}  (average of {len(SEEDS)} seeds)")
        print(f"{'Method':38s} {'PSNR (dB)':>10s} {'SSIM':>8s}")
        for r in rows:
            if r["severity"] == severity:
                print(f"{r['method']:38s} {r['psnr_db']:10.2f} {r['ssim']:8.4f}")


def save_chart(rows, path):
    try:
        import matplotlib
        matplotlib.use("Agg")  # draw to a file, no window needed
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib is not installed, so the chart was skipped. "
              "Run: pip install matplotlib")
        return False

    methods = list(VARIANTS)
    x = np.arange(len(SEVERITIES))
    width = 0.8 / len(methods)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    for ax, key, title in [(axes[0], "psnr_db", "PSNR (Peak Signal-to-Noise Ratio), higher is better"),
                           (axes[1], "ssim", "SSIM (Structural Similarity Index Measure), higher is better")]:
        for i, name in enumerate(methods):
            vals = [next(r[key] for r in rows if r["severity"] == s and r["method"] == name)
                    for s in SEVERITIES]
            ax.bar(x + (i - (len(methods) - 1) / 2) * width, vals, width, label=name)
        ax.set_xticks(x)
        ax.set_xticklabels(SEVERITIES)
        ax.set_xlabel("Damage severity")
        ax.set_title(title, fontsize=10)
        ax.grid(axis="y", alpha=0.3)
    axes[0].set_ylabel("PSNR (dB)")
    axes[1].set_ylabel("SSIM")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, fontsize=9)
    fig.suptitle(f"Ablation study: average over {len(SEEDS)} random seeds")
    fig.tight_layout(rect=(0, 0.12, 1, 0.96))
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return True


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    rows = run_evaluation()
    print_table(rows)
    csv_path = os.path.join(OUT_DIR, "results.csv")
    png_path = os.path.join(OUT_DIR, "ablation.png")
    save_csv(rows, csv_path)
    print(f"\nsaved {csv_path}")
    if save_chart(rows, png_path):
        print(f"saved {png_path}")


if __name__ == "__main__":
    main()