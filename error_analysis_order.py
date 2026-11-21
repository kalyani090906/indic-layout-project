"""
Error analysis for reading order (asked for by the faculty):
which kinds of pages are hardest, and what kind of mistake each method makes.

Uses the labeled validation + test pages (37 pages; the models are not trained again).
  1. score raster, XY-cut and the pairwise MLP (P2) on every page, with page facts
     (language, single / multi-column, table, figure, number of boxes)     -> order_pages.csv
  2. mean score per page type, also for P1 and the Transformer (test pages) -> order_by_type.csv
  3. every box pair put in the wrong order: which classes, and where the
     two boxes are on the page (same column / side by side)                 -> order_wrong_pairs.csv
  4. pictures of the pages where the MLP or XY-cut goes wrong               -> page_<name>.png

    python error_analysis_order.py
Saves to results/error_analysis/
"""

import csv
import os
from collections import Counter

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image

from read_page import draw_order
from reading_order_utils import (DATA_DIR, load_gt_boxes, load_pair_mlp, load_split_pages, mlp_order, order_scores,
                                 overlap, raster_order, xy_cut)

OUT_DIR = os.path.join("results", "error_analysis")
os.makedirs(OUT_DIR, exist_ok=True)

models, xycut_feature = load_pair_mlp()
METHODS = {"raster": raster_order, "xy-cut": xy_cut, "mlp": lambda boxes: mlp_order(models, boxes, xycut_feature)}
page_info = {p["page"]: p for p in csv.DictReader(open("reading_order/pages.csv", encoding="utf-8"))}


def size_group(n):
    return "2-6 boxes" if n <= 6 else "7-10 boxes" if n <= 10 else "11+ boxes"


def where(a, b):
    """Where two boxes are relative to each other."""
    (ax1, ay1, ax2, ay2), (bx1, by1, bx2, by2) = a["bbox"], b["bbox"]
    if overlap(ax1, ax2, bx1, bx2) > 0.5:
        return "one above the other (same column)"
    if overlap(ay1, ay2, by1, by2) > 0.5:
        return "side by side (different columns)"
    return "diagonal (different column and height)"


def wrong_pairs(true_order, pred_order):
    """Box pairs that the prediction puts in the opposite order to the label."""
    rank = {b["id"]: i for i, b in enumerate(pred_order)}
    out = []
    for i, a in enumerate(true_order):
        for b in true_order[i + 1:]:          # label says: a before b
            if rank[a["id"]] > rank[b["id"]]:
                out.append((a, b))
    return out


# ---- 1. score every page ----
split_pages = load_split_pages()
rows, pair_rows, orders = [], [], {}
for split in ["val", "test"]:
    for p in split_pages[split]:
        info = page_info[p["page"]]
        row = dict(page=p["page"], split=split, language=p["language"], columns=p["columns"],
                   table=info["table"], figure=info["figure"], boxes=len(p["boxes"]),
                   size=size_group(len(p["boxes"])))
        for name, fn in METHODS.items():
            pred = fn(p["boxes"])
            orders[(p["page"], name)] = pred
            row[name] = round(order_scores(p["boxes"], pred)["kendall_tau"], 3)
            if name != "raster":
                for a, b in wrong_pairs(p["boxes"], pred):
                    pair_rows.append(dict(method=name, page=p["page"], columns=p["columns"],
                                          first=a["class"], second=b["class"], where=where(a, b)))
        rows.append(row)
pages = pd.DataFrame(rows)
pages.to_csv(os.path.join(OUT_DIR, "order_pages.csv"), index=False)
pd.DataFrame(pair_rows).to_csv(os.path.join(OUT_DIR, "order_wrong_pairs.csv"), index=False)

# ---- 2. mean score per page type ----
# P1 and the Transformer (R2) only saved test-page scores, so the table is on the 21 test pages
test = pages[pages.split == "test"].copy()
for run, col in [("P1_mlp", "P1"), ("R2_pretrain_finetune", "transformer")]:
    saved = pd.read_csv(os.path.join("runs_order", run, "predictions.csv")).set_index("page")["kendall_tau"]
    test[col] = test.page.map(saved).round(3)

groups = [("all pages", lambda d: d)]
for col in ["columns", "language", "size", "table", "figure"]:
    for value in sorted(test[col].unique()):
        groups.append((f"{col} = {value}", lambda d, c=col, v=value: d[d[c] == v]))
by_type = []
for label, pick in groups:
    for name, data in [("test (21 pages)", test), ("val + test (37 pages)", pages)]:
        g = pick(data)
        row = {"pages": name, "group": label, "n": len(g)}
        for m in ["raster", "xy-cut", "mlp", "P1", "transformer"]:
            if m in g:
                row[m] = round(g[m].mean(), 3)
                row[m + "_perfect"] = int((g[m] == 1).sum())
        by_type.append(row)
by_type = pd.DataFrame(by_type)
by_type.to_csv(os.path.join(OUT_DIR, "order_by_type.csv"), index=False)
print(by_type[by_type.pages == "test (21 pages)"].drop(columns="pages").to_string(index=False))
print()
print(by_type[by_type.pages != "test (21 pages)"][["group", "n", "raster", "xy-cut", "mlp", "mlp_perfect",
                                                     "xy-cut_perfect"]].to_string(index=False))

# ---- 3. what kind of mistakes ----
pairs = pd.DataFrame(pair_rows)
for m in ["xy-cut", "mlp"]:
    d = pairs[pairs.method == m]
    print(f"\n{m}: {len(d)} wrong pairs on {d.page.nunique()} of {len(pages)} pages")
    print(d["where"].value_counts().to_string())
    print(Counter(zip(d["first"], d["second"])).most_common(6))

# ---- 4. pictures of pages with mistakes: your label | MLP | XY-cut ----
bad = pages[(pages.mlp < 1) | (pages["xy-cut"] < 1)]
for _, r in bad.iterrows():
    image = Image.open(os.path.join(DATA_DIR, "images", r.page)).convert("RGB")
    all_boxes = load_gt_boxes(r.page)
    label = split_pages[r.split]
    label = next(p for p in label if p["page"] == r.page)["boxes"]
    panels = [("Your label", label), (f"MLP (tau {r.mlp:.2f})", orders[(r.page, "mlp")]),
              (f"XY-cut (tau {r['xy-cut']:.2f})", orders[(r.page, "xy-cut")])]
    fig, axes = plt.subplots(1, 3, figsize=(15, 7.5))
    for ax, (title, order) in zip(axes, panels):
        ax.imshow(draw_order(image, all_boxes, order))
        ax.set_title(title, fontsize=13)
        ax.axis("off")
    fig.suptitle(f"{r.page}  ({r.language}, {r.columns}-column, {r.split})", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.95])   # leave room for the title
    fig.savefig(os.path.join(OUT_DIR, "page_" + r.page.replace("/", "_")), dpi=80)
    plt.close(fig)
print(f"\n{len(bad)} page pictures and the tables saved to {OUT_DIR}/")
