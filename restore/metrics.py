"""Evaluation: PSNR and SSIM need a clean reference image; OCR is optional."""
import os
import shutil

import numpy as np
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

# Where the Windows installer puts Tesseract, used when it is not on PATH.
_WINDOWS_TESSERACT = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
_warned = False


def psnr_ssim(restored: np.ndarray, reference: np.ndarray) -> dict:
    h, w = min(restored.shape[0], reference.shape[0]), min(restored.shape[1], reference.shape[1])
    r, g = restored[:h, :w], reference[:h, :w]
    return {"PSNR_dB": round(float(peak_signal_noise_ratio(g, r)), 2),
            "SSIM": round(float(structural_similarity(g, r)), 4)}


def tesseract_available(quiet: bool = False) -> bool:
    """True if pytesseract is installed AND the Tesseract program can be found.

    Prints one clear message (instead of crashing) when it cannot be used.
    """
    global _warned
    try:
        import pytesseract
    except ImportError:
        problem = "the Python package pytesseract is not installed (run: pip install pytesseract)"
    else:
        cmd = shutil.which("tesseract")
        if cmd is None and os.path.exists(_WINDOWS_TESSERACT):
            cmd = _WINDOWS_TESSERACT
        if cmd is not None:
            pytesseract.pytesseract.tesseract_cmd = cmd
            return True
        problem = ("the Tesseract program was not found. Install it with: "
                   "winget install --id UB-Mannheim.TesseractOCR -e   then run: "
                   '$env:Path += ";C:\\Program Files\\Tesseract-OCR"')
    if not quiet and not _warned:
        print(f"OCR (Optical Character Recognition) skipped: {problem}")
        _warned = True
    return False


def ocr_text(img: np.ndarray) -> str:
    """Read the text in an image. Returns '' if Tesseract is unavailable or fails."""
    if not tesseract_available():
        return ""
    try:
        import pytesseract
        return pytesseract.image_to_string(img, config="--psm 6")
    except Exception as e:  # keep the evaluation running, but say what happened
        print(f"OCR failed on one image: {e}")
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


def _normalize(text: str) -> str:
    """Collapse all line breaks and repeated spaces so layout does not count as errors."""
    return " ".join(text.split())


def evaluate_ocr(degraded: np.ndarray, restored: np.ndarray, truth_text: str = None) -> dict:
    """CER of the OCR text before (degraded) and after (restored) restoration.

    truth_text defaults to the known text of the synthetic page (degrade.DEFAULT_LINES).
    Returns {'cer_before', 'cer_after', 'text_before', 'text_after'}, or None when
    Tesseract is not available.
    """
    if not tesseract_available():
        return None
    if truth_text is None:
        from degrade import DEFAULT_LINES
        truth_text = "\n".join(DEFAULT_LINES)
    truth = _normalize(truth_text)
    before, after = _normalize(ocr_text(degraded)), _normalize(ocr_text(restored))
    return {"cer_before": char_error_rate(before, truth),
            "cer_after": char_error_rate(after, truth),
            "text_before": before, "text_after": after}


if __name__ == "__main__":
    # Run from the project folder with:  python -m restore.metrics
    from degrade import make_clean_page, degrade
    from restore.pipeline import restore

    if tesseract_available():
        clean = make_clean_page()
        base = evaluate_ocr(clean, clean)
        print(f"Sanity check, OCR on the clean page: CER = {base['cer_before']}  (should be near 0)")
        print(f"\n{'Severity':10s} {'CER before':>11s} {'CER after':>10s}   (lower is better, seed 0)")
        for severity in ("light", "medium", "heavy"):
            bad = degrade(clean, 0, severity)
            res = evaluate_ocr(bad, restore(bad)[0])
            print(f"{severity:10s} {res['cer_before']:11.4f} {res['cer_after']:10.4f}")