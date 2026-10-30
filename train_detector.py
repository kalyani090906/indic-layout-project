"""
Layout detector study (YOLO, Ultralytics): how much does the starting point matter, and which
augmentations help on document pages?

Transfer study (all YOLOv8n unless noted, same data, same epochs):
  D0  released YOLOv10x IndicDLP checkpoint, evaluated only (reference)
  D1  from scratch (random weights)
  D2  COCO pre-trained (everyday photos), full fine-tuning
  D3  COCO pre-trained, backbone frozen (only neck + head learn)
  D4  DocLayNet pre-trained (English documents), full fine-tuning
  D5  DocLayNet pre-trained, backbone frozen
  D6  YOLOv8s (bigger), COCO pre-trained, full fine-tuning

Augmentation study (on the best transfer setup):
  default   Ultralytics defaults: mosaic, left-right flip, colour jitter, scale, translate
  none      no augmentation at all
  noflip    defaults without left-right flip (mirrored text is not a real page)
  nomosaic  defaults without mosaic (mosaic glues 4 pages into one image)
  doc       document-style: no flip, no mosaic, small scale / shift / rotation, light brightness change

Examples:
  python train_detector.py --name D0_released --model yolov10x_indicdlp.pt --eval-only
  python train_detector.py --name D1_scratch --model yolov8n.yaml
  python train_detector.py --name D3_coco_frozen --model yolov8n.pt --freeze 10
  python train_detector.py --name A1_none --model yolov8n-doclaynet.pt --aug none

Each run saves to runs_det/<name>/ (Ultralytics files + loss_curve.png + test metrics)
and adds one row to runs_det/summary.csv.
"""

import argparse
import csv
import os
import time

import matplotlib
matplotlib.use("Agg")  # save plots to files, no window
import matplotlib.pyplot as plt
import pandas as pd
from ultralytics import YOLO

parser = argparse.ArgumentParser()
parser.add_argument("--name", required=True)
parser.add_argument("--model", required=True, help="weights (.pt) to start from, or a .yaml for random weights")
parser.add_argument("--data", default="data.yaml")
parser.add_argument("--freeze", type=int, default=0, help="freeze the first N layers (10 = YOLOv8 backbone)")
parser.add_argument("--aug", default="default", choices=["default", "none", "noflip", "nomosaic", "doc"])
parser.add_argument("--epochs", type=int, default=40)
parser.add_argument("--patience", type=int, default=10)
parser.add_argument("--imgsz", type=int, default=640)
parser.add_argument("--batch", type=int, default=32)
parser.add_argument("--device", default="0")
parser.add_argument("--workers", type=int, default=4)
parser.add_argument("--cache", default="ram", help="ram / disk / none")
parser.add_argument("--fraction", type=float, default=1.0, help="use only this share of the train images (quick test)")
parser.add_argument("--eval-only", action="store_true", help="only evaluate --model on the test split")
args = parser.parse_args()

PROJECT = os.path.abspath("runs_det")
OUT_DIR = os.path.join(PROJECT, args.name)

AUGMENT = {
    "default": {},
    "none": dict(mosaic=0.0, fliplr=0.0, hsv_h=0.0, hsv_s=0.0, hsv_v=0.0, translate=0.0, scale=0.0, erasing=0.0),
    "noflip": dict(fliplr=0.0),
    "nomosaic": dict(mosaic=0.0),
    "doc": dict(mosaic=0.0, fliplr=0.0, hsv_h=0.0, hsv_s=0.0, hsv_v=0.3, translate=0.05, scale=0.2, degrees=1.0),
}


def count_params(model, freeze):
    """Total parameters, and the ones that are trained (layers from index `freeze` on)."""
    layers = model.model.model
    total = sum(p.numel() for p in layers.parameters())
    trainable = sum(p.numel() for i, layer in enumerate(layers) if i >= freeze for p in layer.parameters())
    return total, trainable


def plot_losses(results_csv):
    """Train vs val loss per epoch (box + class + distribution-focal loss added up)."""
    df = pd.read_csv(results_csv)
    df.columns = [c.strip() for c in df.columns]
    parts = ["box_loss", "cls_loss", "dfl_loss"]
    train = df[[f"train/{p}" for p in parts]].sum(axis=1)
    val = df[[f"val/{p}" for p in parts]].sum(axis=1)
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    ax[0].plot(df["epoch"], train, marker="o", label="train")
    ax[0].plot(df["epoch"], val, marker="o", label="val")
    ax[0].set_xlabel("epoch"); ax[0].set_ylabel("loss (box + cls + dfl)"); ax[0].legend(); ax[0].set_title("loss")
    ax[1].plot(df["epoch"], df["metrics/mAP50-95(B)"], marker="o", color="green")
    ax[1].set_xlabel("epoch"); ax[1].set_ylabel("val mAP50-95"); ax[1].set_title("validation mAP50-95")
    fig.suptitle(args.name)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "loss_curve.png"), dpi=120)
    return df


model = YOLO(args.model)
row = dict(name=args.name, model=os.path.basename(args.model), freeze=args.freeze, aug=args.aug, imgsz=args.imgsz)

# ============================================================
# 1. Train (early stopping on the val split)
# ============================================================
if not args.eval_only:
    start = time.time()
    model.train(data=args.data, epochs=args.epochs, patience=args.patience, imgsz=args.imgsz, batch=args.batch,
                device=args.device, workers=args.workers, cache=False if args.cache == "none" else args.cache,
                freeze=args.freeze if args.freeze else None, pretrained=not args.model.endswith(".yaml"),
                fraction=args.fraction, project=PROJECT, name=args.name, exist_ok=True, seed=0, plots=True,
                **AUGMENT[args.aug])
    minutes = (time.time() - start) / 60
    df = plot_losses(os.path.join(OUT_DIR, "results.csv"))
    best = df["metrics/mAP50-95(B)"].idxmax()
    row.update(epochs_run=len(df), best_epoch=int(df.loc[best, "epoch"]),
               val_map50_95=round(float(df.loc[best, "metrics/mAP50-95(B)"]), 4),
               min_per_epoch=round(minutes / len(df), 2))
    model = YOLO(os.path.join(OUT_DIR, "weights", "best.pt"))

# counted on the final model (its head now has the 42 IndicDLP classes)
total_params, trainable_params = count_params(model, args.freeze)
row.update(total_params=total_params, trainable_params=0 if args.eval_only else trainable_params)

# ============================================================
# 2. Test (the 500 test pages, never used for training or early stopping)
# ============================================================
metrics = model.val(data=args.data, split="test", imgsz=args.imgsz, batch=args.batch, device=args.device.split(",")[0],
                    project=PROJECT, name=args.name + "_test", exist_ok=True, plots=True)
row.update(test_precision=round(float(metrics.box.mp), 4), test_recall=round(float(metrics.box.mr), 4),
           test_map50=round(float(metrics.box.map50), 4), test_map50_95=round(float(metrics.box.map), 4))

# per-class test mAP50-95, to see which region types each setup finds hard
names = model.names
with open(os.path.join(PROJECT, args.name + "_test", "per_class_map.csv"), "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["class", "map50_95"])
    for i, c in enumerate(metrics.box.ap_class_index):
        w.writerow([names[int(c)], round(float(metrics.box.maps[int(c)]), 4)])

summary_path = os.path.join(PROJECT, "summary.csv")
fields = ["name", "model", "freeze", "aug", "imgsz", "total_params", "trainable_params", "epochs_run", "best_epoch",
          "val_map50_95", "test_precision", "test_recall", "test_map50", "test_map50_95", "min_per_epoch"]
new_file = not os.path.exists(summary_path)
with open(summary_path, "a", newline="") as f:
    w = csv.DictWriter(f, fieldnames=fields)
    if new_file:
        w.writeheader()
    w.writerow({k: row.get(k, "") for k in fields})
print({k: row.get(k, "") for k in fields})
