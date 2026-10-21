\# Report Outline: Historical Document Restoration



\## 1. Introduction

\- Old documents suffer from fading, stains, noise, uneven lighting and blur, which makes them hard to read and to digitize.

\- Our project restores such scans using classical Computer Vision (CV) with OpenCV (Open Source Computer Vision Library), with no training data needed.

\- A web app lets anyone upload a scan and download the restored result.



\## 2. Objectives

\- Build a pipeline that measures the damage first and then picks the right steps (smart routing).

\- Improve image quality, measured by PSNR (Peak Signal-to-Noise Ratio) and SSIM (Structural Similarity Index Measure).

\- Improve machine readability, measured by CER (Character Error Rate) from OCR (Optical Character Recognition).

\- Deliver an easy web interface with before/after comparison and downloads.



\## 3. Methodology

\- Analysis: blur score, contrast, background unevenness and noise level are measured from the grayscale image.

\- Restoration steps: deskew, background/stain/fold flattening, non-local-means denoising, level normalization, CLAHE (Contrast Limited Adaptive Histogram Equalization), sharpening, optional Sauvola binarization.

\- Routing: denoise runs only if noisy, CLAHE only if faded, sharpen only if blurry.

\- Evaluation: a synthetic damage generator (degrade.py) makes clean/degraded pairs; an ablation table (evaluate.py) shows what each step contributes.

\- App: Streamlit interface with comparison slider, detection scores, metrics and two downloads (my part).



\## 4. Results

\- Table: PSNR, SSIM and CER for degraded input, partial pipelines and the full pipeline (fill in from evaluate.py).

\- Screenshots of the slider, scores and OCR before/after (report\\screenshots).

\- Short discussion: which step helped most, and the real-image test results from data\\real.



\## 5. Limitations

\- Heavy tears, ink bleed-through and true page dewarping are not handled.

\- A faint fold line can remain.

\- OCR depends on Tesseract and works best on printed text.

\- Synthetic damage is not identical to real aging, so real-image results are judged visually and by OCR.



\## 6. Future Work

\- Dewarping curved pages and inpainting tears.

\- Bleed-through removal.

\- Learned restoration models compared against this classical baseline.

\- Batch processing of whole archives and support for handwriting.

