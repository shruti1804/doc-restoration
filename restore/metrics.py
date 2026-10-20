"""Evaluation: PSNR and SSIM need a clean reference image; OCR is optional."""
import numpy as np
from skimage.metrics import peak_signal_noise_ratio, structural_similarity


def psnr_ssim(restored: np.ndarray, reference: np.ndarray) -> dict:
    h, w = min(restored.shape[0], reference.shape[0]), min(restored.shape[1], reference.shape[1])
    r, g = restored[:h, :w], reference[:h, :w]
    return {"PSNR_dB": round(float(peak_signal_noise_ratio(g, r)), 2),
            "SSIM": round(float(structural_similarity(g, r)), 4)}


def ocr_text(img: np.ndarray) -> str:
    try:
        import pytesseract
        return pytesseract.image_to_string(img)
    except Exception:
        return ""


def char_error_rate(pred: str, truth: str) -> float:
    """CER (Character Error Rate) = edit distance / length of the true text."""
    a, b = pred.strip(), truth.strip()
    if not b:
        return float("nan")
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return round(prev[-1] / len(b), 4)
