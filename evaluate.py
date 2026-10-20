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


REAL_DIR = os.path.join("data", "real")
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}
MAX_WIDTH = 2000  # very large scans are shrunk to this width, for speed


def _titled(img, text, height=600):
    """Resize a grayscale image to a fixed height and add a white title bar on top."""
    import cv2
    h, w = img.shape[:2]
    img = cv2.resize(img, (max(1, int(w * height / h)), height), interpolation=cv2.INTER_AREA)
    bar = np.full((30, img.shape[1]), 255, np.uint8)
    cv2.putText(bar, text, (8, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.6, 0, 1, cv2.LINE_AA)
    return np.vstack([bar, img])


def run_real_images(folder=REAL_DIR):
    """Run the pipeline on every image in data\\real and save before/after pictures.

    For each image it saves outputs\\real\\NAME_compare.png (original | smart routing |
    all steps forced) and outputs\\real\\NAME_restored.png, prints which steps ran, and
    writes outputs\\real_results.csv.
    """
    import cv2
    paths = sorted(os.path.join(root, f) for root, _, files in os.walk(folder)
                   for f in files if os.path.splitext(f)[1].lower() in IMAGE_EXTENSIONS)
    if not paths:
        print(f"No images found in {folder}. Put .jpg or .png files there and run again.")
        return
    out_dir = os.path.join(OUT_DIR, "real")
    os.makedirs(out_dir, exist_ok=True)
    rows = []
    for path in paths:
        name = os.path.splitext(os.path.basename(path))[0]
        data = np.fromfile(path, np.uint8)  # works even with unusual characters in the path
        img = cv2.imdecode(data, cv2.IMREAD_COLOR) if data.size else None
        if img is None:
            print(f"{name}: could not read this file as an image, skipped")
            continue
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        if gray.shape[1] > MAX_WIDTH:
            scale = MAX_WIDTH / gray.shape[1]
            gray = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        try:
            smart, report = restore(gray, force_all=False)
            forced = restore(gray, force_all=True)[0]
        except Exception as e:
            print(f"{name}: the pipeline crashed: {e!r}")
            rows.append({"image": name, "steps_applied": "CRASH", "contrast_before": "",
                         "contrast_after": "", "analysis": repr(e)})
            continue
        panel = np.hstack([_titled(gray, "original"),
                           _titled(smart, "smart routing: " + ", ".join(report["steps_applied"])[:60]),
                           _titled(forced, "all steps forced")])
        cv2.imwrite(os.path.join(out_dir, f"{name}_compare.png"), panel)
        cv2.imwrite(os.path.join(out_dir, f"{name}_restored.png"), smart)
        row = {"image": name,
               "steps_applied": " + ".join(report["steps_applied"]) or "(none)",
               "contrast_before": round(float(gray.std()), 1),
               "contrast_after": round(float(smart.std()), 1),
               "analysis": str(report["analysis"])}
        rows.append(row)
        print(f"\n{name}  ({gray.shape[1]}x{gray.shape[0]} pixels)")
        print(f"  steps applied : {row['steps_applied']}")
        print(f"  contrast (standard deviation of brightness): "
              f"{row['contrast_before']} -> {row['contrast_after']}")
        print(f"  analysis      : {row['analysis']}")
    if rows:
        save_csv(rows, os.path.join(OUT_DIR, "real_results.csv"))
        print(f"\nsaved pictures to {out_dir} and the table to "
              f"{os.path.join(OUT_DIR, 'real_results.csv')}")


STEP_CHAIN = [
    ("1 deskew", steps.deskew),
    ("2 + background correction", steps.correct_background),
    ("3 + denoise", steps.denoise),
    ("4 + level normalization", steps.normalize_levels),
    ("5 + CLAHE", steps.enhance_contrast),
]


def run_step_by_step():
    """Add the pipeline steps one at a time (same order as pipeline.py) and show
    PSNR and SSIM after each one, averaged over the seeds. The step where SSIM drops
    is the step that hurts."""
    clean = make_clean_page()
    rows = []
    for severity in SEVERITIES:
        labels = ["0 degraded (start)"] + [label for label, _ in STEP_CHAIN]
        scores = {label: [] for label in labels}
        for seed in SEEDS:
            img = degrade(clean, seed, severity)
            m = psnr_ssim(img, clean)
            scores[labels[0]].append((m["PSNR_dB"], m["SSIM"]))
            for label, fn in STEP_CHAIN:
                img = fn(img)
                m = psnr_ssim(img, clean)
                scores[label].append((m["PSNR_dB"], m["SSIM"]))
        print(f"\nSeverity: {severity}  (average of {len(SEEDS)} seeds)")
        print(f"{'After step':28s} {'PSNR (dB)':>10s} {'SSIM':>8s} {'SSIM change':>12s}")
        prev_ssim = None
        for label in labels:
            arr = np.array(scores[label])
            psnr, ssim = float(arr[:, 0].mean()), float(arr[:, 1].mean())
            change = "" if prev_ssim is None else f"{ssim - prev_ssim:+.4f}"
            print(f"{label:28s} {psnr:10.2f} {ssim:8.4f} {change:>12s}")
            rows.append({"severity": severity, "after_step": label, "psnr_db": round(psnr, 2),
                         "ssim": round(ssim, 4), "ssim_change": change})
            prev_ssim = ssim
    os.makedirs(OUT_DIR, exist_ok=True)
    path = os.path.join(OUT_DIR, "step_by_step.csv")
    save_csv(rows, path)
    print(f"\nsaved {path}")


if __name__ == "__main__":
    import sys
    if "--real" in sys.argv:
        run_real_images()
    elif "--steps" in sys.argv:
        run_step_by_step()
    else:
        main()