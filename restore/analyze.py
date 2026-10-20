"""Stage 1: Analysis - measure what is wrong with the document."""
import cv2
import numpy as np


def analyze(gray: np.ndarray) -> dict:
    """Return degradation scores for a grayscale image."""
    blur = cv2.Laplacian(gray, cv2.CV_64F).var()          # low = blurry
    contrast = float(gray.std())                           # low = faded
    # Background unevenness: std-dev of a heavily blurred copy
    bg = cv2.medianBlur(cv2.resize(gray, None, fx=0.25, fy=0.25), 31)
    bg_unevenness = float(bg.std())
    # Noise estimate: difference between image and a median-filtered copy
    noise = float(np.mean(np.abs(gray.astype(np.float32) -
                                 cv2.medianBlur(gray, 3).astype(np.float32))))
    return {
        "blur_score": round(float(blur), 2),
        "contrast_std": round(contrast, 2),
        "background_unevenness": round(bg_unevenness, 2),
        "noise_level": round(noise, 2),
        "is_blurry": blur < 100,
        "is_faded": contrast < 50,
        "is_uneven": bg_unevenness > 12,
        "is_noisy": noise > 3,
    }
