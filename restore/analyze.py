"""Stage 1: Analysis - measure what is wrong with the document."""
import cv2
import numpy as np

_K = np.array([[1, -2, 1], [-2, 4, -2], [1, -2, 1]], np.float32)


def _noise_sigma(gray: np.ndarray) -> float:
    """Noise level (pixel units) measured only on flat paper, so text edges are ignored."""
    g = gray.astype(np.float32)
    lap = np.abs(cv2.filter2D(g, -1, _K))
    grad = np.hypot(cv2.Sobel(g, cv2.CV_32F, 1, 0), cv2.Sobel(g, cv2.CV_32F, 0, 1))
    flat = grad <= np.percentile(grad, 50)
    return float(np.sqrt(np.pi / 2) * lap[flat].mean() / 6)


def analyze(gray: np.ndarray) -> dict:
    """Return degradation scores for a grayscale image."""
    noise = _noise_sigma(gray)                       # measured at original size

    # Measure blur and background at a fixed width so scores compare across images
    h, w = gray.shape
    small = gray if w <= 1000 else cv2.resize(gray, (1000, int(h * 1000 / w)),
                                              interpolation=cv2.INTER_AREA)
    smooth = cv2.GaussianBlur(small, (0, 0), 1.5)          # removes grain first
    blur = float(cv2.Laplacian(smooth, cv2.CV_64F).var())   # low = blurry
    contrast = float(small.std())                           # low = faded
    bg = cv2.medianBlur(cv2.resize(small, None, fx=0.25, fy=0.25), 31)
    bg_unevenness = float(bg.std())

    return {
        "blur_score": round(blur, 2),
        "contrast_std": round(contrast, 2),
        "background_unevenness": round(bg_unevenness, 2),
        "noise_level": round(noise, 2),
        "is_blurry": bool(blur < 100),
        "is_faded": bool(contrast < 50),
        "is_uneven": bool(bg_unevenness > 12),
        "is_noisy": bool(noise > 4),
    }