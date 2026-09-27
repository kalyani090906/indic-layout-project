from datasets import load_dataset
from tqdm import tqdm
import yaml

# Load class ID -> name mapping
with open("indic.yaml", "r") as f:
    class_map = yaml.safe_load(f)

class_names = class_map["names"]

# Load dataset in streaming mode (no full download)
ds = load_dataset("ai4bharat/indicdlp", split="train", streaming=True)

# Shuffle (buffered, since true full shuffle isn't possible while streaming)
ds_shuffled = ds.shuffle(seed=42, buffer_size=2000)

# Pull N examples from the shuffled stream, with a progress bar
N = 500
samples = list(tqdm(ds_shuffled.take(N), total=N, desc="Sampling"))

# Quick sanity check on the first sampled example
example = samples[0]
print(type(example))
print(example.keys())

print(type(example['image']))
print(example['image'].size)

print(example['bboxes'][:3])
print(example['category_ids'][:3])

from collections import Counter

class_counter = Counter()

for sample in samples:
    for cid in sample['category_ids']:
        class_counter[cid] += 1

print("\n--- Class distribution across", N, "sampled images ---")
for cid, count in class_counter.most_common():
    print(f"{class_names[cid]:25s} (id={cid:2d}): {count}")