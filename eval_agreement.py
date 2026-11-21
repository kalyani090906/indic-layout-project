"""
How consistent is the labeling? Compare the first pass with the second pass on the 15 relabel pages
(same person, labels_relabel/ against labels/). This is the ceiling for what a model can be scored on.

    python eval_agreement.py

Output: results/agreement.csv (one row per page) and a summary printed here.
"""

import csv
import glob
import json
import os
import sys

from reading_order_utils import bootstrap_ci, order_scores

sys.stdout.reconfigure(encoding="utf-8")
FIRST, SECOND = os.path.join("reading_order", "labels"), os.path.join("reading_order", "labels_relabel")

rows = []
for path in sorted(glob.glob(os.path.join(SECOND, "*.json"))):
    second = json.load(open(path, encoding="utf-8"))
    first = json.load(open(os.path.join(FIRST, os.path.basename(path)), encoding="utf-8"))
    a = sorted([b for b in first["boxes"] if b["order"] is not None], key=lambda b: b["order"])
    b = sorted([b for b in second["boxes"] if b["order"] is not None], key=lambda b: b["order"])
    ids_a, ids_b = {x["id"] for x in a}, {x["id"] for x in b}
    # the order can only be compared on boxes numbered in both passes
    common_a = [x for x in a if x["id"] in ids_b]
    common_b = [x for x in b if x["id"] in ids_a]
    row = dict(page=second["page"], language=second["language"], boxes_first=len(a), boxes_second=len(b),
               boxes_common=len(common_a), same_boxes=ids_a == ids_b, tau="", pairwise="", exact="",
               first_saved=first["saved_at"], second_saved=second["saved_at"])
    if len(common_a) >= 2:
        scores = order_scores(common_a, common_b)
        row.update(tau=scores["kendall_tau"], pairwise=scores["pairwise_acc"], exact=int(scores["exact_match"]))
    rows.append(row)

with open(os.path.join("results", "agreement.csv"), "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
    w.writeheader()
    w.writerows(rows)

print(f"{'page':<24}{'lang':<7}{'first':>6}{'second':>7}{'common':>7}{'tau':>7}{'exact':>6}")
for r in rows:
    tau = f"{r['tau']:.3f}" if r["tau"] != "" else "-"
    print(f"{r['page']:<24}{r['language']:<7}{r['boxes_first']:>6}{r['boxes_second']:>7}{r['boxes_common']:>7}{tau:>7}{str(r['exact']):>6}")

scored = [r for r in rows if r["tau"] != ""]
taus = [r["tau"] for r in scored]
low, high = bootstrap_ci(taus)
print(f"\n{len(rows)} pages relabeled, {len(scored)} have 2+ boxes numbered in both passes")
print(f"mean tau {sum(taus) / len(taus):.3f} (95% CI {low:.3f}..{high:.3f}), "
      f"identical order on {sum(bool(r['exact']) for r in scored)}/{len(scored)} pages, "
      f"same set of read boxes on {sum(r['same_boxes'] for r in rows)}/{len(rows)} pages")
print("first pass saved", min(r["first_saved"] for r in rows), "to", max(r["first_saved"] for r in rows))
print("second pass saved", min(r["second_saved"] for r in rows), "to", max(r["second_saved"] for r in rows))
