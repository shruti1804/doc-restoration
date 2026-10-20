"""Synthetic degradation generator: turns a clean page into an 'old damaged' page.
Gives you unlimited (damaged, clean) pairs so PSNR/SSIM can be measured.

degrade(clean, seed, severity) accepts severity 'light', 'medium' or 'heavy'.
"""
import os

import cv2
import numpy as np

# The text printed on the synthetic page. Kept at module level so that
# restore/metrics.py can use it as the ground truth for CER (Character Error Rate).
DEFAULT_LINES = [
    "HISTORICAL DOCUMENT RESTORATION", "",
    "This record was written in the year 1887 and",
    "has faded considerably over the decades.",
    "Stains, folds and noise make it hard to read.", "",
    "Computer vision can help recover the text",
    "without changing the original information.",
] * 2

# Strength of every damage type for each severity level.
#   fade   : how much the ink fades toward grey (0 = none, 1 = ink invisible)
#   illum  : strength of the uneven-lighting vignette
#   fold   : brightness multiplier along the fold line (lower = darker line)
#   stains : number of stains,  stain_k : how dark each stain is
#   noise  : standard deviation of Gaussian noise
#   bleed  : strength of the mirrored ink bleed-through from the back of the page
#   blur   : Gaussian blur sigma in pixels (0 = none)
#   sp     : fraction of pixels hit by salt-and-pepper noise
#   tint   : strength of the yellow paper tint (applied in colour, then made grey)
SEVERITY = {
    "light":  dict(fade=0.25, illum=0.10, fold=0.92, stains=2, stain_k=0.15,
                   noise=4,  bleed=0.08, blur=0.6, sp=0.002, tint=0.15),
    "medium": dict(fade=0.45, illum=0.20, fold=0.80, stains=5, stain_k=0.35,
                   noise=9,  bleed=0.15, blur=1.0, sp=0.010, tint=0.30),
    "heavy":  dict(fade=0.65, illum=0.45, fold=0.65, stains=8, stain_k=0.50,
                   noise=16, bleed=0.28, blur=1.8, sp=0.030, tint=0.50),
}


def make_clean_page(text_lines=None, size=(900, 650)):
    h, w = size
    img = np.full((h, w), 255, np.uint8)
    lines = text_lines or DEFAULT_LINES
    y = 70
    for ln in lines:
        cv2.putText(img, ln, (40, y), cv2.FONT_HERSHEY_COMPLEX, 0.7, 0, 1, cv2.LINE_AA)
        y += 38
    return img


def degrade(clean: np.ndarray, seed: int = 0, severity: str = "medium") -> np.ndarray:
    if severity not in SEVERITY:
        raise ValueError(f"severity must be one of {list(SEVERITY)}, got {severity!r}")
    p = SEVERITY[severity]
    rng = np.random.default_rng(seed)
    if clean.ndim == 3:
        clean = cv2.cvtColor(clean, cv2.COLOR_BGR2GRAY)
    img = clean.astype(np.float32)
    h, w = img.shape

    # 1. Fading: ink becomes light grey
    img = 255 - (255 - img) * (1 - p["fade"])

    # 2. Ink bleed-through: faint mirrored text from the back of the page
    mirrored = cv2.flip(clean, 1).astype(np.float32)           # flip left-right
    dx, dy = int(rng.integers(2, 6)), int(rng.integers(-3, 4))  # slight misalignment
    shift = np.float32([[1, 0, dx], [0, 1, dy]])
    mirrored = cv2.warpAffine(mirrored, shift, (w, h), borderValue=255)
    mirrored = cv2.GaussianBlur(mirrored, (0, 0), 1.5)          # ink spreads in the paper
    ink_back = (255 - mirrored) / 255.0                         # 1 where back ink is
    img *= 1 - p["bleed"] * ink_back

    # 3. Yellow paper tint: tint in colour, then convert back to grayscale
    t = p["tint"]
    b, g, r = img * (1 - t), img * (1 - 0.25 * t), img * (1 - 0.05 * t)
    img = cv2.cvtColor(cv2.merge([b, g, r]).astype(np.float32), cv2.COLOR_BGR2GRAY)

    # 4. Uneven illumination (darker toward the edges)
    xs, ys = np.meshgrid(np.linspace(0, 1, w), np.linspace(0, 1, h))
    img *= 1 - p["illum"] * ((xs - 0.3) ** 2 + (ys - 0.4) ** 2)

    # 5. Fold shadow line
    fold = int(w * 0.5)
    img[:, fold - 6:fold + 6] *= p["fold"]

    # 6. Stains
    for _ in range(p["stains"]):
        cx, cy, r_ = rng.integers(0, w), rng.integers(0, h), rng.integers(15, 45)
        blob = np.zeros((h, w), np.float32)
        cv2.circle(blob, (int(cx), int(cy)), int(r_), 1.0, -1)
        blob = cv2.GaussianBlur(blob, (0, 0), r_ / 3)
        img *= 1 - p["stain_k"] * blob

    # 7. Blur (out-of-focus or low-quality scan)
    if p["blur"] > 0:
        img = cv2.GaussianBlur(img, (0, 0), p["blur"])

    # 8. Gaussian noise
    img += rng.normal(0, p["noise"], img.shape)

    # 9. Salt-and-pepper noise: random pure black and pure white speckles
    u = rng.random(img.shape)
    img[u < p["sp"] / 2] = 0
    img[u > 1 - p["sp"] / 2] = 255

    return np.clip(img, 0, 255).astype(np.uint8)


if __name__ == "__main__":
    os.makedirs("samples", exist_ok=True)
    os.makedirs("outputs", exist_ok=True)
    clean = make_clean_page()
    cv2.imwrite(os.path.join("samples", "clean.png"), clean)
    cv2.imwrite(os.path.join("samples", "degraded.png"), degrade(clean, 0, "medium"))
    # Side-by-side preview: clean | light | medium | heavy
    panels = [clean] + [degrade(clean, 0, s) for s in ("light", "medium", "heavy")]
    cv2.imwrite(os.path.join("outputs", "severity_preview.png"), np.hstack(panels))
    print("saved samples/clean.png and samples/degraded.png")
    print("saved outputs/severity_preview.png (clean | light | medium | heavy)")