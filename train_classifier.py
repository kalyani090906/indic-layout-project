"""
Region-crop classifier: "what kind of layout region is this crop?"

One script for every classifier experiment in the project:
  - small CNN from scratch (baseline)
  - ResNet-50 transfer learning: linear probe / partial / full fine-tuning
  - overfitting study: class weights, dropout, BatchNorm, no pool after conv1,
    L1 (lasso), L2 (ridge = weight_decay), augmentation

Each run saves to runs_cls/<name>/:
  best_model.pth, history.csv, loss_curve.png, confusion_matrix.png, report.txt
and adds one row to runs_cls/summary.csv.

Examples:
  python train_classifier.py --model cnn --name C0_cnn
  python train_classifier.py --model cnn --name C0_cnn_noweights --no-class-weights
  python train_classifier.py --model resnet50 --strategy linear --name C1_linear
"""

import argparse
import csv
import os
import random
import time

import matplotlib
matplotlib.use("Agg")  # save plots to files, no window
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.models import resnet50, ResNet50_Weights
from sklearn.metrics import classification_report, confusion_matrix, f1_score


# ============================================================
# SETTINGS (from the command line)
# ============================================================

parser = argparse.ArgumentParser()
parser.add_argument("--name", required=True, help="run name, e.g. C0_cnn")
parser.add_argument("--model", choices=["cnn", "resnet50"], default="cnn")
parser.add_argument("--strategy", choices=["linear", "layer4", "layer3", "layer2", "full"], default="full",
                    help="ResNet only: linear = only fc trains; layerN = unfreeze layerN..layer4 + fc; full = everything")
parser.add_argument("--data", default="crops")
parser.add_argument("--epochs", type=int, default=30)
parser.add_argument("--patience", type=int, default=5, help="early stopping on val loss")
parser.add_argument("--batch-size", type=int, default=64)
parser.add_argument("--lr", type=float, default=None,
                    help="learning rate for the new layers (CNN: all layers). Default: 1e-4 for cnn "
                         "(1e-3 made val loss jump around), 1e-3 for the resnet50 head")
parser.add_argument("--backbone-lr", type=float, default=1e-4, help="ResNet only: learning rate for unfrozen pretrained layers")
parser.add_argument("--img-size", type=int, default=None, help="default: 128 for cnn, 224 for resnet50")
# --- class imbalance ---
parser.add_argument("--no-class-weights", action="store_true", help="plain CrossEntropyLoss (to compare)")
# --- regularization (each one switched on separately) ---
parser.add_argument("--dropout", type=float, default=0.0, help="dropout before the final layer, e.g. 0.5")
parser.add_argument("--l1", type=float, default=0.0, help="lasso: adds l1 * sum(|w|) to the loss")
parser.add_argument("--l2", type=float, default=0.0, help="ridge: weight_decay in the optimizer")
parser.add_argument("--no-bn", action="store_true", help="CNN only: remove BatchNorm")
parser.add_argument("--no-first-pool", action="store_true", help="remove the pooling after the first conv layer")
parser.add_argument("--augment", action="store_true", help="small rotation, brightness/contrast, blur (no flips)")
parser.add_argument("--workers", type=int, default=4)
parser.add_argument("--seed", type=int, default=42)
args = parser.parse_args()

if args.img_size is None:
    args.img_size = 128 if args.model == "cnn" else 224
if args.lr is None:
    args.lr = 1e-4 if args.model == "cnn" else 1e-3

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
OUT_DIR = os.path.join("runs_cls", args.name)


# ============================================================
# DATA
# ============================================================

# CNN from scratch: simple 0.5 normalization (as in class).
# ResNet-50 was pretrained on ImageNet, so it uses ImageNet normalization.
if args.model == "cnn":
    mean, std = [0.5, 0.5, 0.5], [0.5, 0.5, 0.5]
else:
    mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]

test_transform = transforms.Compose([
    transforms.Resize((args.img_size, args.img_size)),
    transforms.ToTensor(),
    transforms.Normalize(mean, std),
])

# No RandomHorizontalFlip: a mirrored page region has mirrored text, so it is not a real example.
augment_transform = transforms.Compose([
    transforms.Resize((args.img_size, args.img_size)),
    transforms.RandomRotation(3, fill=255),                      # scan skew
    transforms.ColorJitter(brightness=0.2, contrast=0.2),         # scan lighting
    transforms.RandomApply([transforms.GaussianBlur(3)], p=0.3),  # blurry scans
    transforms.ToTensor(),
    transforms.Normalize(mean, std),
])


def make_loaders():
    train_dataset = datasets.ImageFolder(os.path.join(args.data, "train"),
                                         augment_transform if args.augment else test_transform)
    val_dataset = datasets.ImageFolder(os.path.join(args.data, "val"), test_transform)
    test_dataset = datasets.ImageFolder(os.path.join(args.data, "test"), test_transform)
    assert train_dataset.classes == val_dataset.classes == test_dataset.classes, "class folders differ between splits"

    kw = dict(batch_size=args.batch_size, num_workers=args.workers, pin_memory=(device.type == "cuda"),
              persistent_workers=args.workers > 0)  # don't restart workers every epoch (slow on Windows)
    return (train_dataset,
            DataLoader(train_dataset, shuffle=True, **kw),
            DataLoader(val_dataset, shuffle=False, **kw),
            DataLoader(test_dataset, shuffle=False, **kw))


# ============================================================
# MODELS
# ============================================================

class CNN(nn.Module):
    """Small CNN trained from scratch: 4 blocks of conv -> BatchNorm -> ReLU -> pool."""

    def __init__(self, num_classes, use_bn=True, first_pool=True, dropout=0.0):
        super(CNN, self).__init__()
        layers = []
        in_ch = 3
        for i, out_ch in enumerate([32, 64, 128, 256]):
            layers.append(nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1))
            if use_bn:
                layers.append(nn.BatchNorm2d(out_ch))
            layers.append(nn.ReLU())
            if i > 0 or first_pool:
                layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
            in_ch = out_ch
        self.features = nn.Sequential(*layers)
        self.pool = nn.AdaptiveAvgPool2d(1)   # works for any input size
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(256, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = self.pool(x)
        x = x.view(x.shape[0], -1)  # flatten
        x = self.dropout(x)
        return self.fc(x)           # logits, no softmax


def create_model(num_classes):
    """Returns (model, optimizer)."""
    if args.model == "cnn":
        model = CNN(num_classes, use_bn=not args.no_bn,
                    first_pool=not args.no_first_pool, dropout=args.dropout).to(device)
        optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.l2)
        return model, optimizer

    # ---- ResNet-50 pretrained on ImageNet ----
    model = resnet50(weights=ResNet50_Weights.DEFAULT)
    if args.no_first_pool:
        model.maxpool = nn.Identity()   # keeps 4x more spatial detail for thin text lines

    # Replace the 1000-class ImageNet layer with our classes
    model.fc = nn.Sequential(nn.Dropout(args.dropout), nn.Linear(model.fc.in_features, num_classes))

    # Freeze everything, then unfreeze according to the strategy
    for param in model.parameters():
        param.requires_grad = False
    unfreeze = {
        "linear": [],
        "layer4": ["layer4"],
        "layer3": ["layer3", "layer4"],
        "layer2": ["layer2", "layer3", "layer4"],
        "full": ["conv1", "bn1", "layer1", "layer2", "layer3", "layer4"],
    }[args.strategy]
    for name in unfreeze:
        for param in getattr(model, name).parameters():
            param.requires_grad = True
    for param in model.fc.parameters():
        param.requires_grad = True

    # New layer learns fast, pretrained layers learn slowly (so we don't erase what they know)
    backbone_params = [p for n, p in model.named_parameters() if p.requires_grad and not n.startswith("fc.")]
    groups = [{"params": model.fc.parameters(), "lr": args.lr}]
    if backbone_params:
        groups.append({"params": backbone_params, "lr": args.backbone_lr})
    optimizer = optim.Adam(groups, weight_decay=args.l2)
    return model.to(device), optimizer


def l1_penalty(model):
    """Lasso term: sum of absolute values of all trainable weights."""
    return sum(p.abs().sum() for n, p in model.named_parameters() if p.requires_grad and "weight" in n)


# ============================================================
# TRAIN / EVALUATE
# ============================================================

def run_epoch(model, loader, criterion, optimizer=None):
    """One pass over the data. Trains if an optimizer is given. Returns loss, accuracy, macro-F1."""
    training = optimizer is not None
    model.train() if training else model.eval()
    total_loss, y_true, y_pred = 0.0, [], []

    with torch.set_grad_enabled(training):
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            if training:
                if args.l1 > 0:
                    loss = loss + args.l1 * l1_penalty(model)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            total_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            y_true.extend(labels.cpu().numpy())
            y_pred.extend(predicted.cpu().numpy())

    acc = 100 * np.mean(np.array(y_true) == np.array(y_pred))
    macro_f1 = f1_score(y_true, y_pred, average="macro")
    return total_loss / len(loader), acc, macro_f1, y_true, y_pred


def plot_curves(history):
    epochs = [h["epoch"] for h in history]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))
    ax1.plot(epochs, [h["train_loss"] for h in history], "o-", label="train loss")
    ax1.plot(epochs, [h["val_loss"] for h in history], "o-", label="val loss")
    ax1.set_xlabel("epoch"); ax1.set_ylabel("loss"); ax1.set_title(f"{args.name}: loss"); ax1.legend()
    ax2.plot(epochs, [h["train_acc"] for h in history], "o-", label="train acc")
    ax2.plot(epochs, [h["val_acc"] for h in history], "o-", label="val acc")
    ax2.plot(epochs, [100 * h["val_macro_f1"] for h in history], "o--", label="val macro-F1 (x100)")
    ax2.set_xlabel("epoch"); ax2.set_ylabel("%"); ax2.set_title(f"{args.name}: accuracy"); ax2.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "loss_curve.png"), dpi=120)
    plt.close(fig)


def plot_confusion(y_true, y_pred, class_names):
    cm = confusion_matrix(y_true, y_pred, normalize="true")  # each row sums to 1
    fig, ax = plt.subplots(figsize=(11, 10))
    ax.imshow(cm, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(len(class_names)), class_names, rotation=90, fontsize=8)
    ax.set_yticks(range(len(class_names)), class_names, fontsize=8)
    ax.set_xlabel("predicted"); ax.set_ylabel("true"); ax.set_title(f"{args.name}: confusion matrix (test, row-normalized)")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "confusion_matrix.png"), dpi=120)
    plt.close(fig)


def main():
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    os.makedirs(OUT_DIR, exist_ok=True)

    train_dataset, train_loader, val_loader, test_loader = make_loaders()
    class_names = train_dataset.classes
    num_classes = len(class_names)

    # ---- Class weights for the imbalance: rare classes count more in the loss ----
    counts = np.bincount(train_dataset.targets, minlength=num_classes)
    if args.no_class_weights:
        criterion = nn.CrossEntropyLoss()
    else:
        class_weights = len(train_dataset) / (num_classes * counts)   # = 1.0 for a perfectly balanced class
        criterion = nn.CrossEntropyLoss(weight=torch.tensor(class_weights, dtype=torch.float32, device=device))

    model, optimizer = create_model(num_classes)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())

    print(f"Device: {device} | run: {args.name} | model: {args.model}"
          + (f" ({args.strategy})" if args.model == "resnet50" else ""))
    print(f"Train/val/test crops: {len(train_dataset)} / {len(val_loader.dataset)} / {len(test_loader.dataset)}, {num_classes} classes")
    print(f"Trainable parameters: {trainable:,} of {total:,}")
    print(f"Class weights: {'off' if args.no_class_weights else 'on'} | dropout={args.dropout} l1={args.l1} l2={args.l2} "
          f"bn={not args.no_bn} first_pool={not args.no_first_pool} augment={args.augment}\n")

    # ---- Training loop with early stopping on val loss ----
    history = []
    best_val_loss = float("inf")
    best_epoch = 0
    epochs_without_improvement = 0
    best_path = os.path.join(OUT_DIR, "best_model.pth")

    for epoch in range(1, args.epochs + 1):
        start = time.time()
        train_loss, train_acc, train_f1, _, _ = run_epoch(model, train_loader, criterion, optimizer)
        val_loss, val_acc, val_f1, _, _ = run_epoch(model, val_loader, criterion)
        seconds = time.time() - start

        history.append(dict(epoch=epoch, train_loss=train_loss, val_loss=val_loss, train_acc=train_acc,
                            val_acc=val_acc, train_macro_f1=train_f1, val_macro_f1=val_f1, seconds=seconds))
        print(f"Epoch [{epoch}/{args.epochs}] Train Loss: {train_loss:.4f} Val Loss: {val_loss:.4f} "
              f"Train Acc: {train_acc:.2f}% Val Acc: {val_acc:.2f}% Val macro-F1: {val_f1:.4f} Time: {seconds:.0f}s")

        # save curves every epoch, so a stopped run still has them
        with open(os.path.join(OUT_DIR, "history.csv"), "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=history[0].keys())
            writer.writeheader(); writer.writerows(history)
        plot_curves(history)

        if val_loss < best_val_loss:
            best_val_loss, best_epoch = val_loss, epoch
            epochs_without_improvement = 0
            torch.save(model.state_dict(), best_path)   # save best model
        else:
            epochs_without_improvement += 1
            print(f"  No improvement for {epochs_without_improvement} epoch(s)")
            if epochs_without_improvement >= args.patience:
                print("  Early stopping triggered!")
                break

    # ---- Load best model and test ----
    model.load_state_dict(torch.load(best_path, map_location=device))
    test_loss, test_acc, test_f1, y_true, y_pred = run_epoch(model, test_loader, criterion)
    report = classification_report(y_true, y_pred, target_names=class_names, digits=4, zero_division=0)
    print(f"\nBest epoch: {best_epoch}\nTest Accuracy: {test_acc:.2f}%  Test macro-F1: {test_f1:.4f}\n")
    print(report)

    with open(os.path.join(OUT_DIR, "report.txt"), "w") as f:
        f.write(f"{vars(args)}\n\nBest epoch: {best_epoch}\nTest accuracy: {test_acc:.2f}%\nTest macro-F1: {test_f1:.4f}\n\n{report}")
    plot_confusion(y_true, y_pred, class_names)

    # ---- One summary row per run, for the results table ----
    best = history[best_epoch - 1]
    row = dict(name=args.name, model=args.model, strategy=args.strategy if args.model == "resnet50" else "-",
               class_weights=not args.no_class_weights, dropout=args.dropout, l1=args.l1, l2=args.l2,
               bn=not args.no_bn, first_pool=not args.no_first_pool, augment=args.augment,
               trainable_params=trainable, best_epoch=best_epoch, epochs_run=len(history),
               train_acc=round(best["train_acc"], 2), val_acc=round(best["val_acc"], 2),
               gap=round(best["train_acc"] - best["val_acc"], 2),
               train_loss=round(best["train_loss"], 4), val_loss=round(best["val_loss"], 4),
               val_macro_f1=round(best["val_macro_f1"], 4),
               test_acc=round(test_acc, 2), test_macro_f1=round(test_f1, 4),
               sec_per_epoch=round(np.mean([h["seconds"] for h in history])))
    summary_path = os.path.join("runs_cls", "summary.csv")
    new_file = not os.path.exists(summary_path)
    with open(summary_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=row.keys())
        if new_file:
            writer.writeheader()
        writer.writerow(row)
    print(f"Saved to {OUT_DIR}/ and added a row to {summary_path}")


if __name__ == "__main__":   # needed on Windows when DataLoader uses workers
    main()
