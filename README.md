# Historical Document Restoration

Computer Vision project: restores faded, stained, noisy, unevenly lit document images.

## Run (3 commands)
    pip install -r requirements.txt
    python degrade.py          # makes samples/clean.png + samples/degraded.png
    streamlit run app.py       # web app (upload -> restore -> compare -> download)

    python evaluate.py         # ablation table (PSNR / SSIM)

## Pipeline
analyze -> deskew -> background + stain + fold flattening -> denoise (non-local means)
-> level normalization -> CLAHE (if faded) -> sharpen (if blurry) -> optional Sauvola binarization

## Files
- restore/analyze.py   : measures blur, contrast, background unevenness, noise
- restore/steps.py     : each restoration step
- restore/pipeline.py  : smart routing (analysis decides which steps run)
- restore/metrics.py   : PSNR, SSIM, OCR, CER
- degrade.py           : synthetic damage generator (paired data for evaluation)
- evaluate.py          : ablation table
- app.py               : Streamlit UI

## Known limitations (state these in your report)
Heavy tears, ink bleed-through and true page dewarping are not handled. A faint fold line can remain.
