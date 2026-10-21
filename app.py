"""Streamlit web app. Run: streamlit run app.py"""
import os
import shutil

import cv2
import numpy as np
import streamlit as st
from PIL import Image  # PIL = Python Imaging Library, used to hand images to the slider
from restore import restore
from restore.metrics import psnr_ssim

SLIDER_ERROR = ""
try:
    from streamlit_image_comparison import image_comparison
    HAS_SLIDER = True
except Exception as e:  # any import problem, not only a missing package
    HAS_SLIDER = False
    SLIDER_ERROR = str(e)

MAX_WIDTH = 2000  # uploads wider than this are shrunk for speed

st.set_page_config(page_title="Historical Document Restoration", layout="wide")
st.title("Historical Document Restoration")
st.caption("Computer Vision pipeline: analyse -> route -> restore -> compare")


# ---------- helpers ----------
def decode_image(data: bytes, flag):
    """Decode uploaded bytes into an image. Returns None if the bytes are not a valid image."""
    if not data:
        return None
    try:
        return cv2.imdecode(np.frombuffer(data, np.uint8), flag)
    except cv2.error:
        return None


def limit_width(image, max_width=MAX_WIDTH):
    """Shrink very wide images (keeps aspect ratio). Returns (image, was_resized)."""
    h, w = image.shape[:2]
    if w <= max_width:
        return image, False
    scale = max_width / w
    return cv2.resize(image, (max_width, int(h * scale)), interpolation=cv2.INTER_AREA), True


def setup_tesseract() -> bool:
    """Find the Tesseract program on Windows. Returns False if it is not installed."""
    if os.environ.get("HDR_NO_TESSERACT"):  # test switch: simulates 'Tesseract missing'
        return False
    try:
        import pytesseract
    except ImportError:
        return False
    exe = shutil.which("tesseract")
    if exe is None:
        for p in (r"C:\Program Files\Tesseract-OCR\tesseract.exe",
                  r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe"):
            if os.path.exists(p):
                exe = p
                break
    if exe is None:
        return False
    pytesseract.pytesseract.tesseract_cmd = exe
    return True


@st.cache_data(show_spinner="Reading text with Tesseract...")
def run_ocr(gray_image, psm=6):
    """OCR with light cleanup: 2x upscale for small text, median blur to remove speckle.
    psm = Page Segmentation Mode (6 = one block of text, 4 = columns, 3 = automatic)."""
    import pytesseract
    g = gray_image
    if g.shape[1] < 1500:
        g = cv2.resize(g, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
    g = cv2.medianBlur(g, 3)
    return pytesseract.image_to_string(g, config=f"--oem 3 --psm {psm}")


@st.cache_data(show_spinner="Restoring the document...")
def run_restore(image, force_all):
    """Run the pipeline twice: enhanced grayscale and binarized. Cached so reruns are instant."""
    gray_out, report = restore(image, binarize=False, force_all=force_all)
    bin_out, _ = restore(image, binarize=True, force_all=force_all)
    if bin_out.mean() < 127:  # make sure it is black text on a white page
        bin_out = 255 - bin_out
    return gray_out, bin_out, report


def png_bytes(image) -> bytes:
    ok, buf = cv2.imencode(".png", image)
    return buf.tobytes()


def show_comparison(original_bgr, restored_gray):
    """Draggable before/after slider; falls back to side-by-side if the package is missing or fails."""
    if HAS_SLIDER:
        try:
            h, w = restored_gray.shape[:2]
            before = cv2.resize(original_bgr, (w, h))  # slider needs both images the same size
            before_img = Image.fromarray(cv2.cvtColor(before, cv2.COLOR_BGR2RGB))
            after_img = Image.fromarray(cv2.cvtColor(restored_gray, cv2.COLOR_GRAY2RGB))
            image_comparison(img1=before_img, img2=after_img,
                             label1="Original", label2="Restored",
                             width=900, make_responsive=True)
            return
        except Exception as e:
            st.warning(f"Comparison slider failed ({e}). Showing side-by-side view instead.")
    c1, c2 = st.columns(2)
    c1.subheader("Original")
    c1.image(cv2.cvtColor(original_bgr, cv2.COLOR_BGR2RGB), use_container_width=True)
    c2.subheader("Restored")
    c2.image(restored_gray, use_container_width=True)


# ---------- sidebar ----------
with st.sidebar:
    st.header("Options")
    binarize = st.checkbox("Show binarized view (black text on white)", value=False)
    force_all = st.checkbox("Force all steps (skip smart routing)", value=False)
    clean_file = st.file_uploader("Optional: clean reference image (for PSNR/SSIM)",
                                  type=["png", "jpg", "jpeg"], key="ref")

# ---------- upload ----------
up = st.file_uploader("Upload a scanned or photographed document",
                      type=["png", "jpg", "jpeg", "tif", "bmp"])
if up is None:
    st.info("Upload an image, or try samples\\degraded.png from the project folder.")
    st.stop()

img = decode_image(up.getvalue(), cv2.IMREAD_COLOR)
if img is None:
    st.error(f"'{up.name}' could not be read as an image. It may be corrupted or not a real "
             "image file. Please upload a PNG, JPG, TIF or BMP document scan.")
    st.stop()

orig_w = img.shape[1]
img, was_resized = limit_width(img)
if was_resized:
    st.caption(f"Large image resized from {orig_w} to {img.shape[1]} pixels wide for speed.")

try:
    gray_out, bin_out, report = run_restore(img, force_all)
except Exception as e:
    st.error(f"Restoration failed: {e}")
    st.stop()

out = bin_out if binarize else gray_out
steps = list(report["steps_applied"]) + (["binarize_Sauvola"] if binarize else [])

# ---------- before / after ----------
st.subheader("Before / After (drag the slider)" if HAS_SLIDER else "Before / After")
if not HAS_SLIDER:
    st.caption("Slider not available, showing side-by-side view. "
               "Fix: pip install streamlit-image-comparison  " + SLIDER_ERROR)
show_comparison(img, out)

# ---------- detection scores ----------
st.subheader("What the system detected")
a = report["analysis"]
cols = st.columns(4)
cols[0].metric("Blur score (low = blurry)", a["blur_score"])
cols[1].metric("Contrast (low = faded)", a["contrast_std"])
cols[2].metric("Background unevenness", a["background_unevenness"])
cols[3].metric("Noise level", a["noise_level"])
st.write("**Steps applied:** " + " -> ".join(steps))

# ---------- metrics vs clean reference ----------
if clean_file is not None:
    ref = decode_image(clean_file.getvalue(), cv2.IMREAD_GRAYSCALE)
    if ref is None:
        st.error("The clean reference file could not be read as an image.")
    else:
        ref = cv2.resize(ref, (out.shape[1], out.shape[0]), interpolation=cv2.INTER_AREA)
        gray_in = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        before, after = psnr_ssim(gray_in, ref), psnr_ssim(out, ref)
        st.subheader("Quality metrics vs clean reference")
        st.table({"": ["Before", "After"],
                  "PSNR (dB)": [before["PSNR_dB"], after["PSNR_dB"]],
                  "SSIM": [before["SSIM"], after["SSIM"]]})

# ---------- OCR ----------
if st.checkbox("Show OCR (Optical Character Recognition) text before/after (needs Tesseract)"):
    if not setup_tesseract():
        st.warning("Tesseract was not found, so text reading is switched off. Install it with: "
                   "winget install --id UB-Mannheim.TesseractOCR -e   then restart this app "
                   "(or add C:\\Program Files\\Tesseract-OCR to your PATH). "
                   "Everything else in the app still works.")
    else:
        o1, o2 = st.columns(2)
        source = o1.selectbox("Image to read for 'After'",
                              ["Enhanced grayscale", "Binarized (black text on white)"])
        psm_label = o2.selectbox("Page layout mode",
                                 ["6 - one block of text", "4 - columns", "3 - automatic"])
        psm = int(psm_label.split(" ")[0])
        after_img = gray_out if source.startswith("Enhanced") else bin_out
        try:
            t1, t2 = st.columns(2)
            t1.text_area("Before", run_ocr(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), psm), height=200)
            t2.text_area("After", run_ocr(after_img, psm), height=200)
        except Exception as e:
            st.error(f"Tesseract was found but failed to read the image: {e}")

# ---------- downloads ----------
st.subheader("Download")
d1, d2 = st.columns(2)
d1.download_button("Download enhanced grayscale image", png_bytes(gray_out),
                   "restored_enhanced_gray.png", "image/png", key="dl_gray")
d2.download_button("Download binarized image (black text on white)", png_bytes(bin_out),
                   "restored_binarized.png", "image/png", key="dl_bin")