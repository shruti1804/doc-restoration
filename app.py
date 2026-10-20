"""Streamlit web app. Run: streamlit run app.py"""
import cv2
import numpy as np
import streamlit as st
from restore import restore
from restore.metrics import psnr_ssim, ocr_text

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

c1, c2 = st.columns(2)
c1.subheader("Original")
c1.image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), use_container_width=True)
c2.subheader("Restored")
c2.image(out, use_container_width=True)

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
