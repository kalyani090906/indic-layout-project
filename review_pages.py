"""
Find the labeled pages that need a second look after two problems in the first labeling pass:

  1. nested boxes: a small box inside a bigger one could not be clicked (fixed in label_tool.html),
     so a small box that should be read kept whatever the XY-cut guess gave it
  2. broken ground truth: a few IndicDLP label files are normalized twice, so every box was squashed
     into the top-left corner (fixed in reading_order_utils.load_gt_boxes). Their saved labels are
     useless, so they are moved to reading_order/labels_broken/ and the page is labeled from scratch.

    python review_pages.py
    python label_tool.py --review      # then check only these pages

Output: reading_order/review_pages.csv  (page, reason)
"""

import csv
import glob
import json
import os
import shutil

from reading_order_utils import ROLES


def area(b):
    return (b[2] - b[0]) * (b[3] - b[1])


def inside(a, b, tol=0.003):
    return a[0] >= b[0] - tol and a[1] >= b[1] - tol and a[2] <= b[2] + tol and a[3] <= b[3] + tol


review = []
os.makedirs("reading_order/labels_broken", exist_ok=True)
for path in sorted(glob.glob("reading_order/labels/*.json")):
    label = json.load(open(path, encoding="utf-8"))
    boxes = [b["bbox"] for b in label["boxes"]]

    if all(area(b) < 1e-4 for b in boxes):
        shutil.move(path, os.path.join("reading_order/labels_broken", os.path.basename(path)))
        review.append({"page": label["page"], "reason": "broken boxes, label again"})
        continue

    # small boxes that should be read but sit inside a bigger box
    nested = [a for a in label["boxes"] if ROLES.get(a["class"], "read") != "skip"
              and any(area(a["bbox"]) < area(b) and inside(a["bbox"], b) for b in boxes)]
    if nested:
        review.append({"page": label["page"], "reason": f"{len(nested)} nested box(es): "
                       + ", ".join(sorted({b['class'] for b in nested}))})

with open("reading_order/review_pages.csv", "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=["page", "reason"])
    w.writeheader()
    w.writerows(review)

broken = sum(r["reason"].startswith("broken") for r in review)
print(f"{len(review)} pages to review ({broken} broken, label again; {len(review) - broken} with nested boxes)")
print("Saved reading_order/review_pages.csv. Now run: python label_tool.py --review")
