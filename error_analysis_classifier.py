"""
Error analysis for the region classifier (asked for by the faculty):
which sections are hardest for the top 3 models, and what they are confused with.

  1. runs each saved model on the test crops (no training) -> predictions_<run>.csv
  2. per-class F1 for the 3 models                          -> per_class_f1.csv
  3. most common mistakes for the hardest classes           -> top_confusions.csv
  4. examples of wrong crops for the hardest classes        -> wrong_<class>.png

    python error_analysis_classifier.py
Saves to results/error_analysis/
"""

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.models import resnet50

# the top 3 classifier runs by test macro-F1
RUNS = {
    "C4_layer2": "runs_cls_downloaded/colab1/C4_layer2",       # partial fine-tuning, F1 0.528
    "C5_full": "runs_cls_downloaded/colab2/C5_full",            # full fine-tuning,    F1 0.518
    "G2b_l2_1e-2": "runs_cls_downloaded/colab2/G2b_l2_1e-2",    # C2 + strong L2,      F1 0.512
}
OUT_DIR = os.path.join("results", "error_analysis")
N_HARD = 5          # how many hardest classes to look at
N_EXAMPLES = 12     # wrong crops shown per hard class

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
os.makedirs(OUT_DIR, exist_ok=True)

# same test transform as train_classifier.py (ResNet-50, ImageNet normalization)
test_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])
test_dataset = datasets.ImageFolder(os.path.join("crops", "test"), test_transform)
class_names = test_dataset.classes
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)   # no workers: this script has no main guard


def load_model(run_dir):
    """Same architecture as in train_classifier.py: ResNet-50 with fc = Dropout + Linear."""
    model = resnet50()
    model.fc = nn.Sequential(nn.Dropout(0.0), nn.Linear(model.fc.in_features, len(class_names)))
    model.load_state_dict(torch.load(os.path.join(run_dir, "best_model.pth"), map_location=device))
    return model.to(device).eval()


def predict(run, run_dir):
    path = os.path.join(OUT_DIR, f"predictions_{run}.csv")
    if os.path.exists(path):   # already done, don't run the model again
        return pd.read_csv(path)
    model = load_model(run_dir)
    preds, confs = [], []
    with torch.no_grad():
        for images, _ in test_loader:
            probs = torch.softmax(model(images.to(device)), dim=1)
            conf, pred = torch.max(probs, 1)
            preds.extend(pred.cpu().numpy())
            confs.extend(conf.cpu().numpy())
    df = pd.DataFrame({
        "file": [p for p, _ in test_dataset.samples],
        "true": [class_names[t] for t in test_dataset.targets],
        "pred": [class_names[p] for p in preds],
        "confidence": np.round(confs, 3),
    })
    df.to_csv(path, index=False)
    return df


# ---- 1. predictions ----
predictions = {}
for run, run_dir in RUNS.items():
    predictions[run] = predict(run, run_dir)
    df = predictions[run]
    print(f"{run}: test acc {100 * (df.true == df.pred).mean():.2f}%, "
          f"macro-F1 {f1_score(df.true, df.pred, average='macro'):.4f}")

# ---- 2. per-class F1 ----
support = predictions["C4_layer2"].true.value_counts()
f1_table = pd.DataFrame({run: f1_score(df.true, df.pred, labels=class_names, average=None)
                         for run, df in predictions.items()}, index=class_names)
f1_table["mean_f1"] = f1_table.mean(axis=1)
f1_table["test_crops"] = support.reindex(class_names).values
f1_table = f1_table.sort_values("mean_f1").round(3)
f1_table.to_csv(os.path.join(OUT_DIR, "per_class_f1.csv"), index_label="class")
print("\nHardest classes (mean F1 over the 3 models):")
print(f1_table.head(N_HARD).to_string())
print("\nEasiest classes:")
print(f1_table.tail(N_HARD).to_string())

# ---- 3. what the hardest classes are confused with (best model, C4) ----
best = predictions["C4_layer2"]
hard = list(f1_table.index[:N_HARD])
rows = []
for cls in hard:
    wrong = best[(best.true == cls) & (best.pred != cls)]
    total = (best.true == cls).sum()
    for pred_cls, n in wrong.pred.value_counts().head(3).items():
        rows.append({"true": cls, "predicted_as": pred_cls, "count": n, "share_of_class": round(n / total, 3)})
    # the other direction: crops of other classes wrongly called this class (low precision)
    false_pos = best[(best.pred == cls) & (best.true != cls)]
    for true_cls, n in false_pos.true.value_counts().head(3).items():
        rows.append({"true": true_cls, "predicted_as": cls, "count": n,
                     "share_of_class": round(n / (best.true == true_cls).sum(), 3)})
pd.DataFrame(rows).to_csv(os.path.join(OUT_DIR, "top_confusions.csv"), index=False)

# do all 3 models make the same mistakes? share of crops wrong in all 3
all_wrong = np.ones(len(best), dtype=bool)
for df in predictions.values():
    all_wrong &= (df.true != df.pred).values
print(f"\nCrops wrong in all 3 models: {all_wrong.sum()} of {len(best)} ({100 * all_wrong.mean():.1f}%)")
for run, df in predictions.items():
    print(f"  wrong in {run}: {(df.true != df.pred).sum()}")

# ---- 4. examples of wrong crops (C4), most confident mistakes first ----
for cls in hard:
    wrong = best[(best.true == cls) & (best.pred != cls)].sort_values("confidence", ascending=False).head(N_EXAMPLES)
    cols = 4
    nrows = int(np.ceil(len(wrong) / cols))
    fig, axes = plt.subplots(nrows, cols, figsize=(cols * 3.2, nrows * 2.4))
    for ax in np.array(axes).ravel():
        ax.axis("off")
    for ax, (_, r) in zip(np.array(axes).ravel(), wrong.iterrows()):
        ax.imshow(Image.open(r.file).convert("RGB"))
        ax.set_title(f"predicted: {r.pred}\n({r.confidence:.2f})", fontsize=9)
    fig.suptitle(f"True class: {cls}  (C4, most confident mistakes)", fontsize=12)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, f"wrong_{cls}.png"), dpi=100)
    plt.close(fig)

print(f"\nSaved to {OUT_DIR}/")
