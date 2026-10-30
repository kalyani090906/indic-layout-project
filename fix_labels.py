"""
Repair the label files that build_subset.py normalized twice (fixed in build_subset.py now).

Some IndicDLP samples already store boxes in 0-1 units; dividing by the image size again squashed every
box of those pages into the top-left corner. This multiplies them back by the image size.
The original files are kept in dataset_root/labels_original/.

    python fix_labels.py

Prints a checksum of all label files, so a copy built elsewhere (e.g. on Kaggle) can be compared.
"""

import glob
import hashlib
import os
import shutil

from PIL import Image

DATA_DIR = "dataset_root"

fixed = 0
for path in sorted(glob.glob(os.path.join(DATA_DIR, "labels", "*", "*.txt"))):
    rows = [line.split() for line in open(path)]
    rows = [p for p in rows if len(p) == 5]
    if not rows or not all(float(p[3]) < 0.01 and float(p[4]) < 0.01 for p in rows):
        continue
    split, name = path.split(os.sep)[-2:]
    W, H = Image.open(os.path.join(DATA_DIR, "images", split, name.replace(".txt", ".png"))).size
    backup = os.path.join(DATA_DIR, "labels_original", split, name)
    os.makedirs(os.path.dirname(backup), exist_ok=True)
    shutil.copy(path, backup)
    with open(path, "w") as f:
        for c, xc, yc, w, h in rows:
            f.write(f"{c} {float(xc) * W} {float(yc) * H} {float(w) * W} {float(h) * H}\n")
    fixed += 1
    print("fixed", split, name)

print(f"{fixed} label files fixed (originals in {DATA_DIR}/labels_original/)")


def label_checksum(data_dir):
    """md5 over all label files, values rounded to 4 decimals (so float formatting does not matter)."""
    digest = hashlib.md5()
    for path in sorted(glob.glob(os.path.join(data_dir, "labels", "*", "*.txt"))):
        digest.update(os.path.basename(path).encode())
        for line in open(path):
            p = line.split()
            if len(p) == 5:
                digest.update((p[0] + " " + " ".join(f"{float(v):.4f}" for v in p[1:]) + ";").encode())
    return digest.hexdigest()


print("label checksum:", label_checksum(DATA_DIR))
