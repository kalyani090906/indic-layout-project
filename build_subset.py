from datasets import load_dataset
from tqdm import tqdm
import yaml
import os

# ---- Config ----
# Each local split is drawn from the matching official IndicDLP split, so our
# val/test pages were never seen by the released checkpoint during training.
SPLITS = {
    # local name: (official HF split, number of images)
    "train": ("train", 4000),
    "val": ("validation", 500),
    "test": ("test", 500),
}
SEED = 42
BUFFER_SIZE = 500
OUT_DIR = "dataset_root"

# ---- Load class mapping ----
with open("indic.yaml", "r") as f:
    class_map = yaml.safe_load(f)
class_names = class_map["names"]

# ---- Save images + YOLO-format labels ----
for split_name, (hf_split, n) in SPLITS.items():
    ds = load_dataset("ai4bharat/indicdlp", split=hf_split, streaming=True)
    ds_shuffled = ds.shuffle(seed=SEED, buffer_size=BUFFER_SIZE)

    os.makedirs(os.path.join(OUT_DIR, "images", split_name), exist_ok=True)
    os.makedirs(os.path.join(OUT_DIR, "labels", split_name), exist_ok=True)

    for i, sample in enumerate(tqdm(ds_shuffled.take(n), total=n, desc=f"Saving {split_name} (from {hf_split})")):
        img = sample["image"]
        img_w, img_h = img.size

        # Save image
        img_filename = f"{split_name}_{i:05d}.png"
        img_path = os.path.join(OUT_DIR, "images", split_name, img_filename)
        img.save(img_path)

        # Convert bboxes to YOLO format and save label file
        label_filename = f"{split_name}_{i:05d}.txt"
        label_path = os.path.join(OUT_DIR, "labels", split_name, label_filename)

        # A few IndicDLP samples already have boxes in 0-1 units instead of pixels.
        # Dividing those by the image size again would squash every box into the top-left corner.
        already_normalized = all(v <= 1.0 for bbox in sample["bboxes"] for v in bbox)
        if already_normalized:
            img_w, img_h = 1.0, 1.0

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

print("Done. Saved", sum(n for _, n in SPLITS.values()), "images across train/val/test.")