"""
Collect every classifier run (laptop, two Colab accounts, Kaggle) into one table,
draw comparison plots, and check whether L1 really makes weights sparse compared to L2.

    python compare_runs.py

Output in results/:
  all_runs.csv            one row per run, from all summary.csv files
  transfer_curves.png     val loss per epoch for CNN + C1-C5 (how much to unfreeze)
  regularization_curves.png  train vs val loss for C2 and every regularizer on top of it
  sparsity.csv            share of near-zero weights in layer4 + fc for C2 / L1 / L2 runs
"""

import csv
import glob
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

RUN_FOLDERS = ["runs_cls", "runs_cls_downloaded/colab1", "runs_cls_downloaded/colab2", "runs_cls_downloaded/kaggle"]
OUT_DIR = "results"
os.makedirs(OUT_DIR, exist_ok=True)

# ============================================================
# 1. Merge all summary.csv files
# ============================================================
runs, run_dir = [], {}
for folder in RUN_FOLDERS:
    path = os.path.join(folder, "summary.csv")
    if not os.path.exists(path):
        print("missing:", path)
        continue
    for row in csv.DictReader(open(path, encoding="utf-8-sig")):
        row["source"] = folder
        runs.append(row)
        run_dir[row["name"]] = os.path.join(folder, row["name"])

with open(os.path.join(OUT_DIR, "all_runs.csv"), "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=list(runs[0].keys()))
    w.writeheader()
    w.writerows(runs)


def history(name):
    rows = list(csv.DictReader(open(os.path.join(run_dir[name], "history.csv"))))
    return {k: [float(r[k]) for r in rows] for k in rows[0]}


def print_table(title, names):
    print(f"\n{title}")
    print(f"{'run':<20}{'best ep':>8}{'train acc':>10}{'val acc':>9}{'gap':>7}"
          f"{'test acc':>10}{'test F1':>9}{'sec/ep':>8}")
    by_name = {r["name"]: r for r in runs}
    for n in names:
        if n not in by_name:
            print(f"{n:<20}  (not found)")
            continue
        r = by_name[n]
        print(f"{n:<20}{r['best_epoch']:>8}{float(r['train_acc']):>10.1f}{float(r['val_acc']):>9.1f}"
              f"{float(r['gap']):>7.1f}{float(r['test_acc']):>10.1f}{float(r['test_macro_f1']):>9.3f}"
              f"{r['sec_per_epoch']:>8}")


CNN_RUNS = ["C0_cnn_lr1e-3", "C0_cnn", "C0_cnn_noweights", "G6_cnn_nobn"]
TRANSFER_RUNS = ["C1_linear", "C2_layer4", "C3_layer3", "C4_layer2", "C5_full"]
REG_RUNS = ["C2_layer4", "G2a_l2_1e-4", "G2b_l2_1e-2", "G3_l1_1e-6", "G3b_l1_1e-5", "G4_elastic",
            "G5_dropout05", "G10_augment", "G7_nopool", "C2_noweights"]

print_table("CNN from scratch (128 px)", CNN_RUNS)
print_table("ResNet-50 transfer learning (how much to unfreeze)", TRANSFER_RUNS)
print_table("Overfitting study on C2 (layer4 unfrozen)", REG_RUNS)

# ============================================================
# 2. Plots
# ============================================================
fig, ax = plt.subplots(figsize=(8, 5))
for n in ["C0_cnn"] + TRANSFER_RUNS:
    h = history(n)
    ax.plot(h["epoch"], h["val_loss"], marker="o", label=n)
ax.set_xlabel("epoch")
ax.set_ylabel("val loss")
ax.set_title("Validation loss: CNN from scratch vs ResNet-50 strategies")
ax.legend()
fig.tight_layout()
fig.savefig(os.path.join(OUT_DIR, "transfer_curves.png"), dpi=120)

reg_plot = [n for n in REG_RUNS if n in run_dir]
cols = 5
rows_n = (len(reg_plot) + cols - 1) // cols
fig, axes = plt.subplots(rows_n, cols, figsize=(4 * cols, 3.2 * rows_n), sharey=True)
for ax, n in zip(axes.flat, reg_plot):
    h = history(n)
    ax.plot(h["epoch"], h["train_loss"], marker="o", label="train")
    ax.plot(h["epoch"], h["val_loss"], marker="o", label="val")
    ax.set_title(n, fontsize=10)
    ax.set_xlabel("epoch")
for ax in axes.flat[len(reg_plot):]:
    ax.axis("off")
axes.flat[0].set_ylabel("loss")
axes.flat[0].legend()
fig.suptitle("Train vs val loss for each regularizer (L1 runs include the L1 penalty in train loss)")
fig.tight_layout()
fig.savefig(os.path.join(OUT_DIR, "regularization_curves.png"), dpi=120)

# ============================================================
# 3. Sparsity check: does L1 push weights to zero more than L2?
# ============================================================
# Only layer4 and fc are trained in these runs, so only those weights are compared.
SPARSITY_RUNS = ["C2_layer4", "G2a_l2_1e-4", "G2b_l2_1e-2", "G3_l1_1e-6", "G3b_l1_1e-5", "G4_elastic"]
print(f"\nSparsity of layer4 + fc weights")
print(f"{'run':<16}{'mean |w|':>10}{'|w|<1e-3':>10}{'|w|<1e-4':>10}")
sparsity_rows = []
for n in SPARSITY_RUNS:
    if n not in run_dir:
        continue
    state = torch.load(os.path.join(run_dir[n], "best_model.pth"), map_location="cpu")
    w = torch.cat([v.flatten().abs() for k, v in state.items()
                   if (k.startswith("layer4") or k.startswith("fc")) and "weight" in k and "bn" not in k
                   and "downsample.1" not in k])
    row = {"run": n, "mean_abs_w": w.mean().item(),
           "frac_below_1e-3": (w < 1e-3).float().mean().item(),
           "frac_below_1e-4": (w < 1e-4).float().mean().item()}
    sparsity_rows.append(row)
    print(f"{n:<16}{row['mean_abs_w']:>10.5f}{100 * row['frac_below_1e-3']:>9.2f}%{100 * row['frac_below_1e-4']:>9.2f}%")

with open(os.path.join(OUT_DIR, "sparsity.csv"), "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(sparsity_rows[0].keys()))
    w.writeheader()
    w.writerows(sparsity_rows)

print(f"\n{len(runs)} runs merged. Saved {OUT_DIR}/all_runs.csv, transfer_curves.png, "
      f"regularization_curves.png, sparsity.csv")
