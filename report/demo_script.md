\# 3-Minute Demo Script



Before starting: run `.venv\\Scripts\\Activate.ps1`, then `$env:Path += ";C:\\Program Files\\Tesseract-OCR"`, then `streamlit run app.py`. Have samples\\degraded.png and samples\\clean.png ready.



\*\*0:00 - 0:25 Intro\*\*

"This is Historical Document Restoration. Old scans are faded, stained and unevenly lit. Our app analyses each image and restores it with classical OpenCV, with no training needed."



\*\*0:25 - 0:50 Upload\*\*

Drag samples\\degraded.png into the uploader. "I upload a degraded scan. The app processes it in a couple of seconds."



\*\*0:50 - 1:20 Detection scores\*\*

Point at the four numbers. "The system measured four things: blur, contrast, background unevenness and noise. A low contrast number means faded; a high unevenness number means stains or bad lighting."



\*\*1:20 - 1:45 Steps applied\*\*

Read the 'Steps applied' line. "Smart routing used these scores to choose steps. A clean image would skip denoising or contrast enhancement." Optionally tick 'Force all steps' to show the difference.



\*\*1:45 - 2:15 Before/after slider\*\*

Drag the slider slowly across the text. "Left is the original, right is restored. Background is flat, text is dark and readable." Tick 'Show binarized view' to show black text on a white page.



\*\*2:15 - 2:45 Metrics and OCR\*\*

Upload samples\\clean.png in the sidebar. "Against the clean reference, PSNR and SSIM both go up." Tick the OCR checkbox. "Tesseract reads far more correct text from the restored image than from the original."



\*\*2:45 - 3:00 Download and close\*\*

Click both download buttons. "Users get the enhanced grayscale image and a binarized version. Limits: heavy tears and bleed-through. Thank you."

