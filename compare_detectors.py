"""
Collect the detector study (Kaggle Part 1 and Part 2) into one table and draw the report plots.

    python compare_detectors.py

D2 (COCO, full fine-tuning, default augmentation) is also the "default" row of the augmentation study.
Both Kaggle runs used the same pages, but not the same pages as the laptop's dataset_root/.

Output in results/:
  detector_runs.csv          one row per run, from both summary.csv files
  detector_scores.png        test mAP50-95 of every run
  detector_transfer_curves.png      train vs val loss for D1-D6
  detector_augmentation_curves.png  train vs val loss for the default setup and A1-A4
"""

import csv
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

RUN_FOLDERS = ["runs_det_downloaded/part1", "runs_det_downloaded/part2"]
OUT_DIR = "results"
os.makedirs(OUT_DIR, exist_ok=True)

LABELS = {
    "D0_released": "D0 released YOLOv10x (evaluated only)",
    "D1_scratch": "D1 YOLOv8n, from scratch",
    "D2_coco_full": "D2 COCO, full (default augmentation)",
    "D3_coco_frozen": "D3 COCO, frozen backbone",
    "D4_doclaynet_full": "D4 DocLayNet, full",
    "D5_doclaynet_frozen": "D5 DocLayNet, frozen backbone",
    "D6_v8s_coco_full": "D6 YOLOv8s, COCO, full",
    "A1_none": "A1 no augmentation",
    "A2_noflip": "A2 no left-right flip",
    "A3_nomosaic": "A3 no mosaic",
    "A4_doc": "A4 document-style",
}
TRAIN_COLOR, VAL_COLOR = "#2a78d6", "#eb6834"
BAR_COLOR, BG, INK, MUTED = "#2a78d6", "#fcfcfb", "#0b0b0b", "#52514e"
plt.rcParams.update({"figure.facecolor": BG, "axes.facecolor": BG, "savefig.facecolor": BG, "text.color": INK,
                     "axes.labelcolor": MUTED, "xtick.color": MUTED, "ytick.color": MUTED, "axes.edgecolor": "#c3c2b7",
                     "axes.spines.top": False, "axes.spines.right": False})

# ============================================================
# 1. Merge the two summary.csv files
# ============================================================
runs, run_dir = [], {}
for folder in RUN_FOLDERS:
    for row in csv.DictReader(open(os.path.join(folder, "summary.csv"))):
        row["source"] = folder
        runs.append(row)
        run_dir[row["name"]] = os.path.join(folder, row["name"])

with open(os.path.join(OUT_DIR, "detector_runs.csv"), "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(runs[0].keys()))
    w.writeheader()
    w.writerows(runs)

print(f"{'run':<22}{'epochs':>7}{'best':>6}{'val mAP':>9}{'test mAP':>10}{'test mAP50':>12}{'params (M)':>12}")
for r in runs:
    print(f"{r['name']:<22}{r['epochs_run']:>7}{r['best_epoch']:>6}{r['val_map50_95']:>9}{r['test_map50_95']:>10}"
          f"{r['test_map50']:>12}{int(r['total_params']) / 1e6:>12.2f}")


def losses(name):
    df = pd.read_csv(os.path.join(run_dir[name], "results.csv"))
    df.columns = [c.strip() for c in df.columns]
    parts = ["box_loss", "cls_loss", "dfl_loss"]
    return df["epoch"], df[[f"train/{p}" for p in parts]].sum(axis=1), df[[f"val/{p}" for p in parts]].sum(axis=1)


# ============================================================
# 2. Test score of every run
# ============================================================
fig, ax = plt.subplots(figsize=(9, 5))
names = [r["name"] for r in runs][::-1]
scores = [float(next(r for r in runs if r["name"] == n)["test_map50_95"]) for n in names]
bars = ax.barh([LABELS[n] for n in names], scores, height=0.62, color=BAR_COLOR)
ax.bar_label(bars, fmt="%.3f", padding=4, color=INK, fontsize=9)
ax.set_xlim(0, 0.62)
ax.set_xlabel("test mAP50-95 (500 pages)")
ax.tick_params(axis="y", length=0)
ax.xaxis.grid(True, color="#e8e7e2")
ax.set_axisbelow(True)
ax.set_title("Detector study: test score of every run", loc="left", fontsize=11)
fig.tight_layout()
fig.savefig(os.path.join(OUT_DIR, "detector_scores.png"), dpi=150)


# ============================================================
# 3. Train vs val loss, one small panel per run, same y-axis
# ============================================================
def loss_panels(names, title, filename):
    fig, axes = plt.subplots(2, 3, figsize=(12, 6.2), sharex=True, sharey=True)
    for ax, n in zip(axes.flat, names):
        epoch, train, val = losses(n)
        ax.plot(epoch, train, color=TRAIN_COLOR, linewidth=2, label="train")
        ax.plot(epoch, val, color=VAL_COLOR, linewidth=2, label="val")
        ax.set_title(LABELS[n], fontsize=10, loc="left")
        ax.yaxis.grid(True, color="#e8e7e2")
        ax.set_ylim(2, 8)
    for ax in axes.flat[len(names):]:
        ax.axis("off")
    for ax in axes[-1]:
        ax.set_xlabel("epoch")
        ax.tick_params(labelbottom=True)
    for ax in axes[:, 0]:
        ax.set_ylabel("loss (box + cls + dfl)")
    axes.flat[0].legend(frameon=False)
    fig.suptitle(title, x=0.01, ha="left", fontsize=12)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, filename), dpi=150)


loss_panels(["D1_scratch", "D2_coco_full", "D3_coco_frozen", "D4_doclaynet_full", "D5_doclaynet_frozen", "D6_v8s_coco_full"],
            "Transfer study: train vs val loss (y-axis cut at 8; D1 starts near 14)", "detector_transfer_curves.png")
loss_panels(["D2_coco_full", "A1_none", "A2_noflip", "A3_nomosaic", "A4_doc"],
            "Augmentation study: without augmentation, val loss rises while train loss keeps falling",
            "detector_augmentation_curves.png")

print(f"\n{len(runs)} runs merged. Saved {OUT_DIR}/detector_runs.csv, detector_scores.png, "
      f"detector_transfer_curves.png, detector_augmentation_curves.png")
