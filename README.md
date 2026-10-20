# Historical Document Restoration

A Computer Vision (CV) project that restores faded, stained, noisy and unevenly lit scans of old documents using classical OpenCV (Open Source Computer Vision Library) image processing. No deep learning is used: a smart router measures each image first and runs only the steps it needs.

## Features
- Automatic damage analysis: blur, contrast, background unevenness, noise
- Smart routing: the analysis decides which restoration steps run
- Draggable before/after comparison slider (with a side-by-side fallback)
- Two downloads: enhanced grayscale image and binarized (black text on white page) image
- OCR (Optical Character Recognition) text before and after, using Tesseract
- Quality metrics: PSNR (Peak Signal-to-Noise Ratio) and SSIM (Structural Similarity Index Measure) against a clean reference

## Screenshots
![Upload and before/after slider](report/screenshots/slider.png)
![Detection scores and steps applied](report/screenshots/scores.png)
![OCR before and after](report/screenshots/ocr.png)

## Setup (Windows PowerShell)
Requirements: Git and Python 3.10 or newer.

    git clone <REPO-URL>
    cd doc-restoration
    python -m venv .venv
    .venv\Scripts\Activate.ps1
    python -m pip install --upgrade pip
    pip install -r requirements.txt

If PowerShell says running scripts is disabled, run this once and activate again:

    Set-ExecutionPolicy -Scope CurrentUser RemoteSigned

Install Tesseract (needed only for the OCR view) and add it to PATH for the current window:

    winget install --id UB-Mannheim.TesseractOCR -e
    $env:Path += ";C:\Program Files\Tesseract-OCR"
    tesseract --version

## Run

    python degrade.py          # makes samples\clean.png and samples\degraded.png
    streamlit run app.py       # web app at http://localhost:8501
    python evaluate.py         # ablation table (PSNR / SSIM)

In the app, upload samples\degraded.png. Optionally upload samples\clean.png in the sidebar to see PSNR and SSIM.

## Pipeline (plain text)

    Input image (BGR or grayscale)
          |
          v
    Convert to grayscale
          |
          v
    ANALYZE: blur score, contrast, background unevenness, noise
          |
          v
    deskew
          |
          v
    background + stain + fold flattening
          |
          v
    denoise (non-local means)        <- only if noisy (or Force all)
          |
          v
    level normalization
          |
          v
    CLAHE (Contrast Limited Adaptive Histogram Equalization)   <- only if faded (or Force all)
          |
          v
    sharpen                          <- only if blurry
          |
          v
    Sauvola binarization             <- only if requested
          |
          v
    Output: restored grayscale image + report (analysis, steps applied)

## Results (to be filled in from evaluate.py)

| Method | PSNR (dB) | SSIM | CER (Character Error Rate) |
|---|---|---|---|
| Degraded input | TBD | TBD | TBD |
| Background flattening only | TBD | TBD | TBD |
| + Denoise | TBD | TBD | TBD |
| Full pipeline | TBD | TBD | TBD |

## Project files
- restore\analyze.py   : measures blur, contrast, background unevenness, noise
- restore\steps.py     : each restoration step
- restore\pipeline.py  : smart routing (analysis decides which steps run)
- restore\metrics.py   : PSNR, SSIM, OCR, CER
- degrade.py           : synthetic damage generator (paired data for evaluation)
- evaluate.py          : ablation table
- app.py               : Streamlit web app
- samples\             : generated sample images
- data\real\           : real test images (not stored in Git)

## Known limitations
- Heavy tears, ink bleed-through and true page dewarping are not handled.
- A faint fold line can remain after flattening.
- Uploads wider than 2000 pixels are shrunk for speed, so very fine detail is reduced.
- Binarization quality depends on how even the lighting is after flattening.
- OCR accuracy depends on Tesseract and works best on printed, not handwritten, text.

## Team
| Member | Role |
|---|---|
| Shruti  | Restoration algorithms: restore\analyze.py, steps.py, pipeline.py |
| Archita | Data and evaluation: degrade.py, evaluate.py, metrics |
| Swapnil | App, OCR and report: app.py, README, demo |