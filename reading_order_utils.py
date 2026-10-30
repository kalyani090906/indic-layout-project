"""
Shared code for the reading-order part: loading page boxes, class roles, and the two rule-based
baselines (raster order and recursive XY-cut).

A box is a dict: {"id": int, "class": str, "bbox": [x1, y1, x2, y2]}  (coordinates normalized to 0-1)
"""

import json
import os
import yaml
from PIL import Image

DATA_DIR = "dataset_root"
NAMES = yaml.safe_load(open("data.yaml"))["names"]
ROLES = yaml.safe_load(open("class_roles.yaml"))["roles"]


def load_gt_boxes(page_id):
    """Ground-truth boxes of a page like 'test/test_00012.png', skipping broken label lines."""
    split, image_name = page_id.split("/")
    label_path = os.path.join(DATA_DIR, "labels", split, image_name.replace(".png", ".txt"))
    rows = [line.split() for line in open(label_path)]
    rows = [[float(v) for v in p] for p in rows if len(p) == 5]

    # A few IndicDLP label files are normalized twice (every value divided by the image size again),
    # so all boxes collapse into the top-left corner. Undo it by multiplying by the image size.
    if rows and all(p[3] < 0.01 and p[4] < 0.01 for p in rows):
        W, H = Image.open(os.path.join(DATA_DIR, "images", page_id)).size
        rows = [[p[0], p[1] * W, p[2] * H, p[3] * W, p[4] * H] for p in rows]

    boxes = []
    for p in rows:
        xc, yc, w, h = p[1:]
        if not (0 <= xc <= 1 and 0 <= yc <= 1 and 0 < w <= 1 and 0 < h <= 1):
            continue
        boxes.append({"id": len(boxes), "class": NAMES[int(p[0])],
                      "bbox": [max(0, xc - w / 2), max(0, yc - h / 2), min(1, xc + w / 2), min(1, yc + h / 2)]})
    return boxes


def is_read(box):
    """True if the box gets a reading-order number (role read or special), False if it is skipped."""
    return ROLES.get(box["class"], "read") != "skip"


# ============================================================
# BASELINE 1: raster order (top to bottom, then left to right)
# ============================================================

def raster_order(boxes):
    return sorted(boxes, key=lambda b: (round(b["bbox"][1], 2), b["bbox"][0]))


# ============================================================
# BASELINE 2: recursive XY-cut
# ============================================================
# Find the widest empty gap, either horizontal (split into top / bottom) or vertical
# (split into left / right columns), cut there, and repeat inside each part.
# The widest gap wins because a column gutter is usually wider than the space between paragraphs.
# When no cut exists, fall back to raster order.

def _gaps(intervals):
    """Empty gaps between 1-D intervals, as (gap_size, cut_position)."""
    intervals = sorted(intervals)
    gaps, reach = [], intervals[0][1]
    for start, end in intervals[1:]:
        if start > reach:
            gaps.append((start - reach, (start + reach) / 2))
        reach = max(reach, end)
    return gaps


def xy_cut(boxes, min_gap=0.002):
    if len(boxes) <= 1:
        return list(boxes)
    h_gaps = [g for g in _gaps([(b["bbox"][1], b["bbox"][3]) for b in boxes]) if g[0] >= min_gap]
    v_gaps = [g for g in _gaps([(b["bbox"][0], b["bbox"][2]) for b in boxes]) if g[0] >= min_gap]
    best_h = max(h_gaps) if h_gaps else (0, None)
    best_v = max(v_gaps) if v_gaps else (0, None)
    if best_h[0] == 0 and best_v[0] == 0:
        return raster_order(boxes)

    if best_h[0] >= best_v[0]:
        cut = best_h[1]   # top part is read before the bottom part
        first = [b for b in boxes if (b["bbox"][1] + b["bbox"][3]) / 2 < cut]
        second = [b for b in boxes if (b["bbox"][1] + b["bbox"][3]) / 2 >= cut]
    else:
        cut = best_v[1]   # left column is read before the right column
        first = [b for b in boxes if (b["bbox"][0] + b["bbox"][2]) / 2 < cut]
        second = [b for b in boxes if (b["bbox"][0] + b["bbox"][2]) / 2 >= cut]
    if not first or not second:
        return raster_order(boxes)
    return xy_cut(first, min_gap) + xy_cut(second, min_gap)


# ============================================================
# Loading labels and scoring a predicted order
# ============================================================

def load_label(page_id, folder="reading_order/labels"):
    """Your saved label: the numbered boxes in reading order (skipped boxes left out)."""
    path = os.path.join(folder, page_id.replace("/", "_").replace(".png", ".json"))
    boxes = json.load(open(path, encoding="utf-8"))["boxes"]
    return sorted([b for b in boxes if b["order"] is not None], key=lambda b: b["order"])


def order_scores(true_order, pred_order):
    """Compare two orders of the same boxes.

    pairwise accuracy: share of box pairs (a, b) put in the same before/after order as the label
    kendall tau:       the same thing rescaled to -1 (fully reversed) .. +1 (identical)
    exact match:       1 if the whole page order is identical
    """
    rank = {b["id"]: i for i, b in enumerate(pred_order)}
    ids = [b["id"] for b in true_order]
    pairs = agree = 0
    for i in range(len(ids)):
        for j in range(i + 1, len(ids)):
            pairs += 1
            agree += rank[ids[i]] < rank[ids[j]]
    pairwise = agree / pairs
    return {"pairwise_acc": pairwise, "kendall_tau": 2 * pairwise - 1, "exact_match": float(agree == pairs)}


def bootstrap_ci(values, n=2000, seed=0):
    """95% confidence interval of a mean, by resampling pages with replacement."""
    import random
    rng = random.Random(seed)
    means = sorted(sum(rng.choice(values) for _ in values) / len(values) for _ in range(n))
    return means[int(0.025 * n)], means[int(0.975 * n)]


# ============================================================
# Shared by the ordering models (train_pairwise.py, train_transformer.py)
# ============================================================
CLASS_INDEX = {c: i for i, c in NAMES.items()}


def box_features(b):
    """x1, y1, x2, y2, centre x, centre y, width, height, class one-hot (8 + 42 numbers)."""
    x1, y1, x2, y2 = b["bbox"]
    onehot = [0.0] * len(NAMES)
    onehot[CLASS_INDEX[b["class"]]] = 1.0
    return [x1, y1, x2, y2, (x1 + x2) / 2, (y1 + y2) / 2, x2 - x1, y2 - y1] + onehot


def load_split_pages():
    """Your labeled pages by split; pages with fewer than 2 numbered boxes are left out."""
    import csv
    split_pages = {"train": [], "val": [], "test": []}
    for p in csv.DictReader(open("reading_order/pages.csv", encoding="utf-8")):
        boxes = load_label(p["page"])
        if len(boxes) >= 2:
            split_pages[p["split"]].append(dict(p, boxes=boxes))
    return split_pages


def save_results(out_dir, name, model, setting, test_pages, all_test, all_val, histories):
    """Loss plot, per-page test predictions (mean over seeds) vs XY-cut, printed table,
    and one row in runs_order/summary.csv. all_test / all_val: one list of page scores per seed."""
    import csv
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    h = histories[0]   # curves of the first seed
    ep = [r["epoch"] for r in h]
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    for i, (metric, label) in enumerate([("loss", "BCE loss"), ("pair_acc", "pair accuracy")]):
        ax[i].plot(ep, [r["train_" + metric] for r in h], label="train")
        ax[i].plot(ep, [r["val_" + metric] for r in h], label="val")
        ax[i].set_xlabel("epoch"); ax[i].set_ylabel(label); ax[i].legend()
    fig.suptitle(name)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "loss_curve.png"), dpi=120)
    plt.close(fig)
    with open(os.path.join(out_dir, "history.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(h[0].keys()))
        w.writeheader()
        w.writerows(h)

    seeds = len(all_test)
    rows = []
    for i, p in enumerate(test_pages):
        row = dict(page=p["page"], language=p["language"], columns=p["columns"], boxes=len(p["boxes"]))
        for m in ["pairwise_acc", "kendall_tau", "exact_match"]:
            row[m] = sum(run[i][m] for run in all_test) / seeds
        row["xycut_tau"] = order_scores(p["boxes"], xy_cut(p["boxes"]))["kendall_tau"]
        rows.append(row)
    with open(os.path.join(out_dir, "predictions.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    mean = lambda rs, k: sum(r[k] for r in rs) / len(rs)
    multi = [r for r in rows if r["columns"] == "multi"]
    diff = [r["kendall_tau"] - r["xycut_tau"] for r in rows]
    lo, hi = bootstrap_ci(diff)
    seed_taus = [mean(run, "kendall_tau") for run in all_test]
    val_tau = sum(mean(run, "kendall_tau") for run in all_val) / seeds

    print(f"\nTEST ({len(rows)} pages, mean of {seeds} seeds)")
    print(f"{'':16}{'model':>8}{'xy-cut':>8}")
    print(f"{'tau all':<16}{mean(rows, 'kendall_tau'):>8.3f}{mean(rows, 'xycut_tau'):>8.3f}")
    print(f"{'tau multi-col':<16}{mean(multi, 'kendall_tau'):>8.3f}{mean(multi, 'xycut_tau'):>8.3f}   ({len(multi)} pages)")
    print(f"{'exact match':<16}{100 * mean(rows, 'exact_match'):>7.0f}%")
    print(f"seed spread of test tau: {min(seed_taus):.3f} .. {max(seed_taus):.3f}")
    print(f"model minus XY-cut tau: {sum(diff) / len(diff):+.3f}  (95% CI {lo:+.3f} .. {hi:+.3f}; "
          f"{'a real difference' if lo > 0 or hi < 0 else 'includes 0, so not a clear difference'})")

    summary_path = os.path.join("runs_order", "summary.csv")
    row = dict(name=name, model=model, setting=setting, seeds=seeds,
               epochs_run=round(sum(len(x) for x in histories) / len(histories), 1), val_tau=round(val_tau, 4),
               test_tau=round(mean(rows, "kendall_tau"), 4), test_tau_multi=round(mean(multi, "kendall_tau"), 4),
               test_pairwise=round(mean(rows, "pairwise_acc"), 4), test_exact=round(mean(rows, "exact_match"), 4),
               xycut_test_tau=round(mean(rows, "xycut_tau"), 4), diff_ci_low=round(lo, 4), diff_ci_high=round(hi, 4))
    new_file = not os.path.exists(summary_path)
    with open(summary_path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(row.keys()))
        if new_file:
            w.writeheader()
        w.writerow(row)
    print(f"Saved {out_dir}/ and a row in {summary_path}")


# ============================================================
# Pairwise MLP (trained by train_pairwise.py, used by read_page.py)
# ============================================================

def overlap(a1, a2, b1, b2):
    """Overlap of two 1-D intervals, as a share of the shorter one (0 = none, 1 = full)."""
    return max(0.0, min(a2, b2) - max(a1, b1)) / max(1e-6, min(a2 - a1, b2 - b1))


def pair_features(a, b, xy_rank, xycut_feature):
    fa, fb = box_features(a), box_features(b)
    pair = [fb[4] - fa[4], fb[5] - fa[5],
            overlap(a["bbox"][0], a["bbox"][2], b["bbox"][0], b["bbox"][2]),
            overlap(a["bbox"][1], a["bbox"][3], b["bbox"][1], b["bbox"][3])]
    if xycut_feature:
        pair.append(1.0 if xy_rank[a["id"]] < xy_rank[b["id"]] else 0.0)
    return fa + fb + pair


def page_pairs(boxes, xycut_feature):
    """All ordered pairs (a, b), a != b. Label 1 if a comes before b (only if the boxes have an 'order')."""
    xy_rank = {b["id"]: i for i, b in enumerate(xy_cut(boxes))}
    X, y, index = [], [], []
    for i, a in enumerate(boxes):
        for j, b in enumerate(boxes):
            if i != j:
                X.append(pair_features(a, b, xy_rank, xycut_feature))
                y.append(1.0 if a.get("order", 0) < b.get("order", 0) else 0.0)
                index.append((i, j))
    return X, y, index


def create_pair_mlp(n_features, hidden, dropout):
    import torch.nn as nn
    return nn.Sequential(
        nn.Linear(n_features, hidden), nn.ReLU(), nn.Dropout(dropout),
        nn.Linear(hidden, hidden), nn.ReLU(), nn.Dropout(dropout),
        nn.Linear(hidden, 1),
    )


def mlp_order(model, boxes, xycut_feature):
    """Score every pair, make it consistent (P(a,b) = 1 - P(b,a)), sort boxes by total 'before' score.
    model can also be a list of models (one per seed): their probabilities are averaged."""
    import numpy as np
    import torch
    if len(boxes) < 2:
        return list(boxes)
    X, _, index = page_pairs(boxes, xycut_feature)
    models = model if isinstance(model, list) else [model]
    X = torch.tensor(X, dtype=torch.float32)
    with torch.no_grad():
        p = np.mean([torch.sigmoid(m.eval()(X).squeeze(1)).numpy() for m in models], axis=0)
    n = len(boxes)
    P = np.zeros((n, n))
    for (i, j), prob in zip(index, p):
        P[i, j] = prob
    P = (P + (1 - P.T)) / 2
    np.fill_diagonal(P, 0)
    return [boxes[i] for i in np.argsort(-P.sum(axis=1), kind="stable")]


def load_pair_mlp(path="runs_order/P2_mlp_xycut/model.pt"):
    import torch
    """All seeds of a trained MLP, as a list (mlp_order averages them)."""
    saved = torch.load(path)
    models = []
    for state in saved["states"]:
        model = create_pair_mlp(saved["n_features"], saved["hidden"], saved["dropout"])
        model.load_state_dict(state)
        models.append(model.eval())
    return models, saved["xycut_feature"]
