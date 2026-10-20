"""Stages 2-7: individual restoration steps. Each takes and returns a grayscale uint8 image."""
import cv2
import numpy as np
from skimage.filters import threshold_sauvola


def deskew(gray: np.ndarray) -> np.ndarray:
    """Straighten a slightly rotated page using the text pixels' minimum-area rectangle."""
    inv = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
    coords = np.column_stack(np.where(inv > 0))
    if len(coords) < 100:
        return gray
    angle = cv2.minAreaRect(coords)[-1]
    angle = -(90 + angle) if angle < -45 else -angle
    if abs(angle) < 0.3 or abs(angle) > 15:
        return gray
    h, w = gray.shape
    M = cv2.getRotationMatrix2D((w // 2, h // 2), angle, 1.0)
    return cv2.warpAffine(gray, M, (w, h), flags=cv2.INTER_CUBIC,
                          borderMode=cv2.BORDER_REPLICATE)


def correct_background(gray: np.ndarray, kernel: int = 9) -> np.ndarray:
    """Estimate the paper layer (yellowing, shadows, fold lines, stains) and divide it out.
    A SMALL closing kernel removes only the thin dark text strokes, so the estimate keeps
    everything that is part of the damaged paper. Dividing by it flattens the page while
    the text (which is not in the estimate) is preserved."""
    se = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel, kernel))
    bg = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, se)
    bg = cv2.medianBlur(bg, 21)
    bg = cv2.GaussianBlur(bg, (0, 0), 5)
    return cv2.divide(gray, np.maximum(bg, 1), scale=255)


def remove_stains(gray: np.ndarray) -> np.ndarray:
    """Detect large blobs that are darker than the page but NOT thin text strokes, then inpaint."""
    blur = cv2.GaussianBlur(gray, (0, 0), 3)
    local = cv2.medianBlur(blur, 61)
    diff = cv2.subtract(local, blur)                       # positive where darker than surroundings
    mask = (diff > 18).astype(np.uint8) * 255
    # Keep only blobs: opening removes thin text strokes
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k)
    mask = cv2.dilate(mask, k)
    if mask.sum() == 0:
        return gray
    return cv2.inpaint(gray, mask, 5, cv2.INPAINT_TELEA)


def denoise(gray: np.ndarray, strength: int = 10) -> np.ndarray:
    """Non-local means denoising."""
    return cv2.fastNlMeansDenoising(gray, None, h=strength,
                                    templateWindowSize=7, searchWindowSize=21)


def enhance_contrast(gray: np.ndarray, clip: float = 2.0) -> np.ndarray:
    """CLAHE (Contrast Limited Adaptive Histogram Equalization)."""
    return cv2.createCLAHE(clipLimit=clip, tileGridSize=(8, 8)).apply(gray)


def sharpen(gray: np.ndarray, amount: float = 1.0) -> np.ndarray:
    """Unsharp mask: boosts faded or blurry text edges."""
    blur = cv2.GaussianBlur(gray, (0, 0), 2)
    return cv2.addWeighted(gray, 1 + amount, blur, -amount, 0)


def binarize(gray: np.ndarray, window: int = 25) -> np.ndarray:
    """Sauvola adaptive thresholding -> clean black text on white page."""
    t = threshold_sauvola(gray, window_size=window)
    return ((gray > t) * 255).astype(np.uint8)


def normalize_levels(gray: np.ndarray, low_pct: float = 1.0) -> np.ndarray:
    """Stretch so the page background becomes white and the darkest ink becomes black."""
    lo = np.percentile(gray, low_pct)
    hi = np.percentile(gray, 70)   # page background dominates the image
    out = (gray.astype(np.float32) - lo) / max(hi - lo, 1) * 255
    return np.clip(out, 0, 255).astype(np.uint8)
