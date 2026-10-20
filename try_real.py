"""Runs restore() on every image in data/real and saves side-by-side results to outputs/."""
import os
import cv2
from restore import restore

os.makedirs("outputs", exist_ok=True)
for name in os.listdir("data/real"):
    if not name.lower().endswith((".png", ".jpg", ".jpeg", ".tif", ".bmp")):
        continue
    img = cv2.imread(os.path.join("data/real", name))
    out, report = restore(img)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    cv2.imwrite(os.path.join("outputs", "cmp_" + name),
                cv2.hconcat([gray, out]))
    print(name, report["steps_applied"], report["analysis"])