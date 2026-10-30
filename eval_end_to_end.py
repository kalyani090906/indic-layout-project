"""
Reading order on DETECTED boxes (what the real pipeline sees), not on the ground-truth boxes.

For each labeled val / test page:
  1. run the layout detector (results are cached in results/detections/)
  2. order the detected boxes with XY-cut and with the pairwise MLP
  3. match each box you numbered to a detected box (IoU >= 0.5, one to one)
  4. compare the order of the matched boxes with your order (Kendall tau)

Also reports how many of your numbered boxes the detector found at all (box recall).

    python eval_end_to_end.py
"""

import csv
import json
import os

from PIL import Image

from read_page import detect_boxes
from reading_order_utils import (DATA_DIR, is_read, load_pair_mlp, load_split_pages, mlp_order, order_scores,
                                 xy_cut)

CACHE = os.path.join("results", "detections")
os.makedirs(CACHE, exist_ok=True)


def iou(a, b):
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0


def match(label_boxes, detected):
    """Greedy one-to-one matching by IoU. Returns {label box id: detected box id}."""
    pairs = sorted(((iou(l["bbox"], d["bbox"]), l["id"], d["id"]) for l in label_boxes for d in detected), reverse=True)
    used_l, used_d, out = set(), set(), {}
    for score, li, di in pairs:
        if score < 0.5:
            break
        if li not in used_l and di not in used_d:
            out[li] = di
            used_l.add(li)
            used_d.add(di)
    return out


def detections(page_id):
    path = os.path.join(CACHE, page_id.replace("/", "_").replace(".png", ".json"))
    if not os.path.exists(path):
        boxes = detect_boxes(Image.open(os.path.join(DATA_DIR, "images", page_id)).convert("RGB"))
        json.dump(boxes, open(path, "w"))
    return json.load(open(path))


models, xycut_feature = load_pair_mlp()
METHODS = {"xy-cut": xy_cut, "mlp": lambda boxes: mlp_order(models, boxes, xycut_feature)}

split_pages = load_split_pages()
rows = []
for split in ["val", "test"]:
    for p in split_pages[split]:
        detected = [b for b in detections(p["page"]) if is_read(b)]
        m = match(p["boxes"], detected)
        row = dict(page=p["page"], split=split, language=p["language"], columns=p["columns"],
                   labeled=len(p["boxes"]), found=len(m), extra=len(detected) - len(m))
        matched_label = [b for b in p["boxes"] if b["id"] in m]   # already in your order
        for name, fn in METHODS.items():
            gt_tau = order_scores(p["boxes"], fn(p["boxes"]))["kendall_tau"]
            if len(matched_label) >= 2:
                det_rank = {b["id"]: k for k, b in enumerate(fn(detected))}
                # your order of the matched boxes vs the method's order of their detected twins
                pred = sorted(matched_label, key=lambda b: det_rank[m[b["id"]]])
                det_tau = order_scores(matched_label, pred)["kendall_tau"]
            else:
                det_tau = None
            row[f"{name}_gt_boxes"] = gt_tau
            row[f"{name}_detected"] = det_tau
        rows.append(row)
        print(".", end="", flush=True)
print()

with open("results/end_to_end.csv", "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
    w.writeheader()
    w.writerows(rows)


def mean(rs, key):
    vals = [r[key] for r in rs if r[key] is not None]
    return sum(vals) / len(vals)


print(f"\n{'split':<6}{'group':<14}{'pages':>6}{'box recall':>11}{'extra':>7}"
      f"{'xy-cut gt':>11}{'xy-cut det':>11}{'mlp gt':>8}{'mlp det':>9}")
for split in ["val", "test"]:
    for group, keep in [("all", lambda r: True), ("multi-column", lambda r: r["columns"] == "multi")]:
        g = [r for r in rows if r["split"] == split and keep(r)]
        recall = sum(r["found"] for r in g) / sum(r["labeled"] for r in g)
        print(f"{split:<6}{group:<14}{len(g):>6}{100 * recall:>10.0f}%{sum(r['extra'] for r in g) / len(g):>7.1f}"
              f"{mean(g, 'xy-cut_gt_boxes'):>11.3f}{mean(g, 'xy-cut_detected'):>11.3f}"
              f"{mean(g, 'mlp_gt_boxes'):>8.3f}{mean(g, 'mlp_detected'):>9.3f}")
print("\nbox recall = share of your numbered boxes the detector found (IoU >= 0.5); "
      "extra = detected read boxes per page with no match")
print("Saved results/end_to_end.csv")
