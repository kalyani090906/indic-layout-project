"""
Make the audio clips for the listening test (rated in the demo, tab "Listening test").

Pages: labeled val / test pages where the MLP and XY-cut read the DETECTED boxes in a different order,
up to 3 Hindi + 3 Tamil. Each page gets two clips, one per method, called "A" and "B" in random order,
so the listener does not know which is which. To keep clips short, only the first MAX_BOXES boxes are
spoken, and only the first WORDS_PER_BOX words of each box: enough to hear where the reading jumps.

    python make_rating_clips.py

Output: outputs/rating_clips/<clip>.wav, outputs/rating_clips/<page>.png (plain page, no numbers),
        outputs/rating_clips/clips.csv (which clip is which method; the demo does not show this)
"""

import csv
import json
import os
import random
import sys

import soundfile as sf
from PIL import Image

from read_page import ocr_box, order_boxes, speak
from reading_order_utils import DATA_DIR, load_split_pages

MAX_BOXES = 8
WORDS_PER_BOX = 15
PER_LANGUAGE = 3
OUT_DIR = os.path.join("outputs", "rating_clips")
os.makedirs(OUT_DIR, exist_ok=True)
sys.stdout.reconfigure(encoding="utf-8")
random.seed(0)

split_pages = load_split_pages()
candidates = [p for s in ["test", "val"] for p in split_pages[s]]
random.shuffle(candidates)

clips, chosen = [], {"hindi": 0, "tamil": 0}
for p in candidates:
    if chosen[p["language"]] >= PER_LANGUAGE:
        continue
    det_path = os.path.join("results", "detections", p["page"].replace("/", "_").replace(".png", ".json"))
    if not os.path.exists(det_path):
        continue   # run eval_end_to_end.py first
    boxes = json.load(open(det_path))
    orders = {m: order_boxes(boxes, m)[:MAX_BOXES] for m in ["mlp", "xy-cut"]}
    if [b["id"] for b in orders["mlp"]] == [b["id"] for b in orders["xy-cut"]]:
        continue   # both methods agree: nothing to compare
    chosen[p["language"]] += 1

    image = Image.open(os.path.join(DATA_DIR, "images", p["page"])).convert("RGB")
    page_name = p["page"].replace("/", "_").replace(".png", "")
    image.save(os.path.join(OUT_DIR, page_name + ".png"))
    texts = {}
    for b in {b["id"]: b for m in orders for b in orders[m]}.values():   # OCR each box once
        texts[b["id"]] = " ".join(ocr_box(image, b, p["language"]).split()[:WORDS_PER_BOX])
    labels = random.sample(["A", "B"], 2)
    for label, method in zip(labels, ["mlp", "xy-cut"]):
        audio, rate = speak([texts[b["id"]] for b in orders[method] if texts[b["id"]]], p["language"])
        clip = f"{page_name}_{label}"
        sf.write(os.path.join(OUT_DIR, clip + ".wav"), audio, rate)
        clips.append(dict(clip=clip, page=p["page"], image=page_name + ".png", language=p["language"],
                          version=label, method=method, boxes=len(orders[method]), seconds=round(len(audio) / rate)))
        print(f"{clip}: {p['language']}, {len(audio) / rate:.0f} s", flush=True)

clips.sort(key=lambda c: (c["language"], c["clip"]))
with open(os.path.join(OUT_DIR, "clips.csv"), "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=list(clips[0].keys()))
    w.writeheader()
    w.writerows(clips)
print(f"{len(clips)} clips from {len(clips) // 2} pages. Saved {OUT_DIR}/clips.csv")
