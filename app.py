"""Streamlit web app. Run: streamlit run app.py"""
import cv2
import numpy as np
import streamlit as st
from PIL import Image  # PIL = Python Imaging Library, used to hand images to the slider
from restore import restore
from restore.metrics import psnr_ssim, ocr_text

try:
    from streamlit_image_comparison import image_comparison
    HAS_SLIDER = True
except ImportError:
    HAS_SLIDER = False

st.set_page_config(page_title="Historical Document Restoration", layout="wide")
st.title("Historical Document Restoration")
st.caption("Computer Vision pipeline: analyse -> route -> restore -> compare")

with st.sidebar:
    st.header("Options")
    binarize = st.checkbox("Binarize output (black text on white)", value=False)
    force_all = st.checkbox("Force all steps (skip smart routing)", value=False)
    clean_file = st.file_uploader("Optional: clean reference image (for PSNR/SSIM)",
                                  type=["png", "jpg", "jpeg"], key="ref")

up = st.file_uploader("Upload a scanned or photographed document", type=["png", "jpg", "jpeg", "tif", "bmp"])
if up is None:
    st.info("Upload an image, or try samples/degraded.png from the project folder.")
    st.stop()

img = cv2.imdecode(np.frombuffer(up.read(), np.uint8), cv2.IMREAD_COLOR)
out, report = restore(img, binarize=binarize, force_all=force_all)


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


st.subheader("Before / After (drag the slider)" if HAS_SLIDER else "Before / After")
show_comparison(img, out)

st.subheader("What the system detected")
a = report["analysis"]
cols = st.columns(4)
cols[0].metric("Blur score (low = blurry)", a["blur_score"])
cols[1].metric("Contrast (low = faded)", a["contrast_std"])
cols[2].metric("Background unevenness", a["background_unevenness"])
cols[3].metric("Noise level", a["noise_level"])
st.write("**Steps applied:** " + " -> ".join(report["steps_applied"]))

if clean_file is not None:
    ref = cv2.imdecode(np.frombuffer(clean_file.read(), np.uint8), cv2.IMREAD_GRAYSCALE)
    gray_in = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    before, after = psnr_ssim(gray_in, ref), psnr_ssim(out, ref)
    st.subheader("Quality metrics vs clean reference")
    st.table({"": ["Before", "After"],
              "PSNR (dB)": [before["PSNR_dB"], after["PSNR_dB"]],
              "SSIM": [before["SSIM"], after["SSIM"]]})

if st.checkbox("Show OCR (Optical Character Recognition) text before/after (needs Tesseract)"):
    t1, t2 = st.columns(2)
    t1.text_area("Before", ocr_text(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)), height=200)
    t2.text_area("After", ocr_text(out), height=200)

ok, buf = cv2.imencode(".png", out)
st.download_button("Download restored image", buf.tobytes(), "restored.png", "image/png")