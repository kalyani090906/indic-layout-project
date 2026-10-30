"""
Cut every labeled box out of the dataset pages and save it as a small image,
so we can train image classifiers ("what kind of region is this?").

Output layout (works directly with torchvision.datasets.ImageFolder):
    crops/train/paragraph/train_00012_3.png
    crops/val/figure/val_00007_0.png
    ...
"""

import os
import random
import csv
import yaml
from collections import Counter, defaultdict
from PIL import Image, ImageOps
from tqdm import tqdm

# ---- Config ----
DATA_DIR = "dataset_root"
OUT_DIR = "crops"
CROP_SIZE = 224            # ResNet-50 input size
MARGIN = 4                 # extra pixels kept around each box
MIN_TRAIN_BOXES = 500      # classes with fewer train boxes are merged into "other"
TRAIN_CAP = 3000           # max crops per class in train (keeps disk + training time sane)
SEED = 42

random.seed(SEED)
names = yaml.safe_load(open("data.yaml"))["names"]


def read_boxes(split):
    """Return a list of (image_name, box_index, class_id, xc, yc, w, h) for one split."""
    boxes = []
    label_dir = os.path.join(DATA_DIR, "labels", split)
    for label_file in sorted(os.listdir(label_dir)):
        if not label_file.endswith(".txt"):
            continue
        image_name = label_file.replace(".txt", ".png")
        for i, line in enumerate(open(os.path.join(label_dir, label_file))):
            parts = line.split()
            if len(parts) != 5:
                continue
            cls = int(parts[0])
            xc, yc, w, h = map(float, parts[1:])
            # skip broken labels (pixel coordinates, negative sizes) found in IndicDLP
            if not (0 <= xc <= 1 and 0 <= yc <= 1 and 0 < w <= 1 and 0 < h <= 1):
                continue
            boxes.append((image_name, i, cls, xc, yc, w, h))
    return boxes


def crop_to_square(page, xc, yc, w, h):
    """Cut the box out, pad it to a white square (no stretching), resize to CROP_SIZE."""
    W, H = page.size
    x1 = max(0, int((xc - w / 2) * W) - MARGIN)
    y1 = max(0, int((yc - h / 2) * H) - MARGIN)
    x2 = min(W, int((xc + w / 2) * W) + MARGIN)
    y2 = min(H, int((yc + h / 2) * H) + MARGIN)
    if x2 - x1 < 2 or y2 - y1 < 2:
        return None
    crop = page.crop((x1, y1, x2, y2))
    side = max(crop.size)
    pad_w, pad_h = side - crop.size[0], side - crop.size[1]
    crop = ImageOps.expand(crop, (pad_w // 2, pad_h // 2, pad_w - pad_w // 2, pad_h - pad_h // 2), fill="white")
    return crop.resize((CROP_SIZE, CROP_SIZE), Image.BILINEAR)


# ---- 1. Decide which classes keep their own folder (based on TRAIN counts only) ----
all_boxes = {split: read_boxes(split) for split in ["train", "val", "test"]}
train_counts = Counter(names[b[2]] for b in all_boxes["train"])
kept_classes = sorted(c for c, n in train_counts.items() if n >= MIN_TRAIN_BOXES)


def folder_name(cls_id):
    name = names[cls_id]
    return name if name in kept_classes else "other"


# ---- 2. Cap train classes, keep val/test as they are (real distribution) ----
selected = {}
for split, boxes in all_boxes.items():
    if split == "train":
        by_class = defaultdict(list)
        for b in boxes:
            by_class[folder_name(b[2])].append(b)
        boxes = []
        for cls, items in by_class.items():
            random.shuffle(items)
            boxes.extend(items[:TRAIN_CAP])
    selected[split] = boxes

# ---- 3. Cut and save crops (open each page once) ----
counts = {split: Counter() for split in selected}
for split, boxes in selected.items():
    by_page = defaultdict(list)
    for b in boxes:
        by_page[b[0]].append(b)

    for image_name, page_boxes in tqdm(by_page.items(), desc=f"Cropping {split}"):
        page = Image.open(os.path.join(DATA_DIR, "images", split, image_name)).convert("RGB")
        for _, i, cls, xc, yc, w, h in page_boxes:
            crop = crop_to_square(page, xc, yc, w, h)
            if crop is None:
                continue
            cls_folder = folder_name(cls)
            out_dir = os.path.join(OUT_DIR, split, cls_folder)
            os.makedirs(out_dir, exist_ok=True)
            crop.save(os.path.join(out_dir, f"{image_name[:-4]}_{i}.png"))
            counts[split][cls_folder] += 1

# ---- 4. Print and save the table of crops per class ----
classes = sorted(set().union(*counts.values()))
with open(os.path.join(OUT_DIR, "crop_counts.csv"), "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["class", "train", "val", "test", "train_before_cap"])
    print(f"\n{'class':<24}{'train':>8}{'val':>8}{'test':>8}{'train(before cap)':>20}")
    for c in sorted(classes, key=lambda c: -counts["train"][c]):
        before = train_counts[c] if c != "other" else sum(n for k, n in train_counts.items() if k not in kept_classes)
        row = [c, counts["train"][c], counts["val"][c], counts["test"][c], before]
        writer.writerow(row)
        print(f"{c:<24}{row[1]:>8}{row[2]:>8}{row[3]:>8}{row[4]:>20}")
    totals = [sum(counts[s].values()) for s in ["train", "val", "test"]]
    print(f"{'TOTAL':<24}{totals[0]:>8}{totals[1]:>8}{totals[2]:>8}")

print(f"\n{len(classes)} classes ({len(kept_classes)} kept + 'other'). Saved to {OUT_DIR}/")
