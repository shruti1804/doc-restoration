"""Synthetic degradation generator: turns a clean page into an 'old damaged' page.
Gives you unlimited (damaged, clean) pairs so PSNR/SSIM can be measured."""
import cv2
import numpy as np


def make_clean_page(text_lines=None, size=(900, 650)):
    h, w = size
    img = np.full((h, w), 255, np.uint8)
    lines = text_lines or [
        "HISTORICAL DOCUMENT RESTORATION", "",
        "This record was written in the year 1887 and",
        "has faded considerably over the decades.",
        "Stains, folds and noise make it hard to read.", "",
        "Computer vision can help recover the text",
        "without changing the original information.",
    ] * 2
    y = 70
    for ln in lines:
        cv2.putText(img, ln, (40, y), cv2.FONT_HERSHEY_COMPLEX, 0.7, 0, 1, cv2.LINE_AA)
        y += 38
    return img


def degrade(clean: np.ndarray, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    img = clean.astype(np.float32)
    h, w = img.shape
    # 1. Fading: ink becomes light grey
    img = 255 - (255 - img) * 0.45
    # 2. Yellow/uneven illumination
    xs, ys = np.meshgrid(np.linspace(0, 1, w), np.linspace(0, 1, h))
    img *= 0.75 + 0.25 * (1 - 0.8 * ((xs - 0.3) ** 2 + (ys - 0.4) ** 2))
    # 3. Fold shadow line
    fold = int(w * 0.5)
    img[:, fold - 6:fold + 6] *= 0.8
    # 4. Stains
    for _ in range(5):
        cx, cy, r = rng.integers(0, w), rng.integers(0, h), rng.integers(15, 45)
        blob = np.zeros((h, w), np.float32)
        cv2.circle(blob, (cx, cy), r, 1.0, -1)
        blob = cv2.GaussianBlur(blob, (0, 0), r / 3)
        img *= 1 - 0.35 * blob
    # 5. Noise
    img += rng.normal(0, 9, img.shape)
    return np.clip(img, 0, 255).astype(np.uint8)


if __name__ == "__main__":
    clean = make_clean_page()
    cv2.imwrite("samples/clean.png", clean)
    cv2.imwrite("samples/degraded.png", degrade(clean))
    print("saved samples/clean.png and samples/degraded.png")
