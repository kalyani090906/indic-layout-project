from datasets import load_dataset
from tqdm import tqdm
import yaml
import os

# ---- Config ----
N_TOTAL = 5000          # start small to test the pipeline; we'll scale this up later
SEED = 42
BUFFER_SIZE = 500
OUT_DIR = "dataset_root"

# ---- Load class mapping ----
with open("indic.yaml", "r") as f:
    class_map = yaml.safe_load(f)
class_names = class_map["names"]

# ---- Load and shuffle the streaming dataset ----
ds = load_dataset("ai4bharat/indicdlp", split="train", streaming=True)
ds_shuffled = ds.shuffle(seed=SEED, buffer_size=BUFFER_SIZE)

# ---- Pull N examples ----
samples = list(tqdm(ds_shuffled.take(N_TOTAL), total=N_TOTAL, desc="Downloading"))

# ---- Split 80:10:10 ----
n_train = int(0.8 * N_TOTAL)
n_val = int(0.1 * N_TOTAL)
# whatever's left goes to test, avoids rounding gaps
splits = {
    "train": samples[:n_train],
    "val": samples[n_train:n_train + n_val],
    "test": samples[n_train + n_val:]
}

# ---- Save images + YOLO-format labels ----
for split_name, split_samples in splits.items():
    for i, sample in enumerate(tqdm(split_samples, desc=f"Saving {split_name}")):
        img = sample["image"]
        img_w, img_h = img.size

        # Save image
        img_filename = f"{split_name}_{i:05d}.png"
        img_path = os.path.join(OUT_DIR, "images", split_name, img_filename)
        img.save(img_path)

        # Convert bboxes to YOLO format and save label file
        label_filename = f"{split_name}_{i:05d}.txt"
        label_path = os.path.join(OUT_DIR, "labels", split_name, label_filename)

        with open(label_path, "w") as lf:
            for bbox, cat_id in zip(sample["bboxes"], sample["category_ids"]):
                x, y, w, h = bbox
                # Convert absolute [x, y, w, h] (top-left corner based)
                # to YOLO's normalized [x_center, y_center, w, h]
                x_center = (x + w / 2) / img_w
                y_center = (y + h / 2) / img_h
                w_norm = w / img_w
                h_norm = h / img_h
                lf.write(f"{cat_id} {x_center} {y_center} {w_norm} {h_norm}\n")

print("Done. Saved", N_TOTAL, "images across train/val/test.")