"""Smart routing: analysis decides which steps run."""
import cv2
import numpy as np
from . import steps
from .analyze import analyze


def restore(image: np.ndarray, binarize: bool = False, force_all: bool = False):
    """
    image: BGR or grayscale uint8 array.
    Returns (restored_image, report) where report lists the analysis and the steps applied.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image.copy()
    report = {"analysis": analyze(gray), "steps_applied": []}
    a = report["analysis"]
    out = gray

    def run(name, fn, *args):
        nonlocal out
        out = fn(out, *args)
        report["steps_applied"].append(name)

    run("deskew", steps.deskew)
    run("background_and_stain_flattening", steps.correct_background)
    if force_all or a["is_noisy"]:
        run("denoise", steps.denoise, int(np.clip(a["noise_level"] * 1.6, 3, 20)))
    run("level_normalization", steps.normalize_levels)
    if force_all or a["is_faded"]:
        run("contrast_enhancement_CLAHE", steps.enhance_contrast)
    if a["is_blurry"]:
        run("sharpen", steps.sharpen)
    if binarize:
        run("binarize_Sauvola", steps.binarize)
    return out, report
