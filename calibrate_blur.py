import os
import cv2
from restore.analyze import analyze

for name in sorted(os.listdir("data/real")):
    if not name.lower().endswith((".png", ".jpg", ".jpeg")):
        continue
    g = cv2.imread(os.path.join("data/real", name), 0)
    for s in [0, 1.5, 3, 5]:
        t = g if s == 0 else cv2.GaussianBlur(g, (0, 0), s)
        print(name, "extra blur sigma", s, "->", analyze(t)["blur_score"])