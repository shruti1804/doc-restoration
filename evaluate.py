"""Run: python evaluate.py  -> ablation table (each stage's contribution) on synthetic data."""
import cv2
import numpy as np
from degrade import make_clean_page, degrade
from restore import steps
from restore.metrics import psnr_ssim
from restore.pipeline import restore

clean = make_clean_page()
rows = []
for seed in range(5):
    bad = degrade(clean, seed)
    variants = {
        "degraded (no processing)": bad,
        "CLAHE only": steps.enhance_contrast(bad),
        "Background correction only": steps.correct_background(bad),
        "Background + denoise": steps.denoise(steps.correct_background(bad)),
        "Full pipeline (smart routing)": restore(bad, force_all=False)[0],
        "Full pipeline (all steps forced)": restore(bad, force_all=True)[0],
    }
    for name, img in variants.items():
        m = psnr_ssim(img, clean)
        rows.append((name, m["PSNR_dB"], m["SSIM"]))

print(f"{'Method':38s} {'PSNR (dB)':>10s} {'SSIM':>8s}")
for name in dict.fromkeys(r[0] for r in rows):
    p = np.mean([r[1] for r in rows if r[0] == name])
    s = np.mean([r[2] for r in rows if r[0] == name])
    print(f"{name:38s} {p:10.2f} {s:8.4f}")
