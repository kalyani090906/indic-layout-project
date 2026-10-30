"""
Choose the pages to label for reading order and split them into train / val / test.

Input:  language_tags.csv (tagged by eye), page_candidates.csv (layout hints)
Output: reading_order/pages.csv  with columns page, language, split, relabel, read_boxes, columns, table, figure

  - 60 Hindi + 60 Tamil pages; pages with < 3 or > 40 boxes to read are left out
  - interesting layouts first (multi-column, table, figure), then pages with many boxes
  - per language: 40 train / 8 val / 12 test, with the interesting layouts spread over the splits
  - relabel = 1 for 15 test pages that you label a second time, about a week later,
    to measure how consistent the labels are (the ceiling for any model)
"""

import csv
import os
import random

FOLDER = "page_selection"   # run from the project folder:  python page_selection/select_pages.py

PER_LANGUAGE = 60
SPLIT_SIZES = {"train": 40, "val": 8, "test": 12}
N_RELABEL = 15
SEED = 42

random.seed(SEED)
hints = {f"{r['split']}/{r['image']}": r for r in csv.DictReader(open(os.path.join(FOLDER, "page_candidates.csv"), encoding="utf-8"))}
tags = list(csv.DictReader(open(os.path.join(FOLDER, "language_tags.csv"), encoding="utf-8")))

chosen = []
for language in ["hindi", "tamil"]:
    pages = []
    for t in tags:
        if t["language"] != language:
            continue
        h = hints[t["page"]]
        boxes = int(h["read_boxes"])
        if 3 <= boxes <= 40:
            pages.append(dict(page=t["page"], language=language, read_boxes=boxes, columns=h["columns"],
                              table=h["has_table"] == "True", figure=h["has_figure"] == "True"))
    random.shuffle(pages)
    interesting = lambda p: (p["columns"] == "multi") + p["table"] + p["figure"]
    pages.sort(key=lambda p: (-interesting(p), -min(p["read_boxes"], 25)))
    pages = pages[:PER_LANGUAGE]

    # deal pages into the splits in "interesting first" order, interleaved,
    # so every split gets some multi-column / table / figure pages
    pages.sort(key=lambda p: (p["columns"] != "multi", -interesting(p), -min(p["read_boxes"], 25)))
    pattern = []
    while len(pattern) < len(pages):
        for s in ["test", "train", "val", "train", "test", "train", "train", "train"]:   # 5:2:1 ~ 40:12:8
            if pattern.count(s) < SPLIT_SIZES[s]:
                pattern.append(s)
    for p, s in zip(pages, pattern):
        p["split"] = s
    chosen.extend(pages)

# 15 test pages (both languages) to label twice
test_pages = [p for p in chosen if p["split"] == "test"]
for p in chosen:
    p["relabel"] = 0
for p in random.sample(test_pages, N_RELABEL):
    p["relabel"] = 1

os.makedirs("reading_order", exist_ok=True)
fields = ["page", "language", "split", "relabel", "read_boxes", "columns", "table", "figure"]
with open("reading_order/pages.csv", "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=fields)
    w.writeheader()
    w.writerows(sorted(chosen, key=lambda p: (p["split"], p["language"], p["page"])))

print(f"{'':14}{'pages':>6}{'multi-col':>10}{'table':>7}{'figure':>8}{'boxes':>7}")
for language in ["hindi", "tamil"]:
    for split in SPLIT_SIZES:
        g = [p for p in chosen if p["language"] == language and p["split"] == split]
        print(f"{language + ' ' + split:<14}{len(g):>6}{sum(p['columns'] == 'multi' for p in g):>10}"
              f"{sum(p['table'] for p in g):>7}{sum(p['figure'] for p in g):>8}{sum(p['read_boxes'] for p in g):>7}")
print(f"\nTotal {len(chosen)} pages, {sum(p['read_boxes'] for p in chosen)} boxes to order, "
      f"{N_RELABEL} test pages marked to relabel. Saved reading_order/pages.csv")
