"""
Score the two rule-based reading-order baselines against your labels.

  raster: top to bottom, then left to right
  xy-cut: recursive cut at the widest empty gap (see reading_order_utils.py)

Both get the boxes you numbered (the skipped ones are left out), so this only measures ORDER.
Pages with fewer than 2 numbered boxes have nothing to order and are left out.

    python eval_baselines.py

Output: results/baselines.csv (one row per page per method) and a table per split and language.
"""

import csv
import os

from reading_order_utils import load_label, order_scores, raster_order, xy_cut

METHODS = {"raster": raster_order, "xy-cut": xy_cut}
METRICS = ["pairwise_acc", "kendall_tau", "exact_match"]

pages = list(csv.DictReader(open("reading_order/pages.csv", encoding="utf-8")))
rows = []
for p in pages:
    true_order = load_label(p["page"])
    if len(true_order) < 2:
        continue
    for method, fn in METHODS.items():
        scores = order_scores(true_order, fn(true_order))
        rows.append(dict(page=p["page"], language=p["language"], split=p["split"], columns=p["columns"],
                         boxes=len(true_order), method=method, **scores))

os.makedirs("results", exist_ok=True)
with open("results/baselines.csv", "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
    w.writeheader()
    w.writerows(rows)


def mean(group, key):
    return sum(r[key] for r in group) / len(group)


print(f"{'split':<7}{'group':<14}{'pages':>6}{'method':>9}{'pairwise':>10}{'tau':>8}{'exact':>8}")
for split in ["train", "val", "test"]:
    for group_name, keep in [("all", lambda r: True), ("hindi", lambda r: r["language"] == "hindi"),
                             ("tamil", lambda r: r["language"] == "tamil"),
                             ("multi-column", lambda r: r["columns"] == "multi")]:
        for method in METHODS:
            g = [r for r in rows if r["split"] == split and r["method"] == method and keep(r)]
            if g:
                print(f"{split:<7}{group_name:<14}{len(g):>6}{method:>9}{mean(g, 'pairwise_acc'):>10.3f}"
                      f"{mean(g, 'kendall_tau'):>8.3f}{100 * mean(g, 'exact_match'):>7.0f}%")
    print()

# the test pages XY-cut gets most wrong: good examples for the report
worst = sorted([r for r in rows if r["split"] == "test" and r["method"] == "xy-cut"], key=lambda r: r["kendall_tau"])[:5]
print("Test pages where XY-cut is worst:")
for r in worst:
    print(f"  {r['page']:<22}{r['language']:<7}{r['columns']:<8}{r['boxes']:>3} boxes   tau {r['kendall_tau']:.2f}")
