"""
Find candidate pages for the reading-order dataset.

For every val + test page (pages the released detector never trained on):
  1. Tesseract script detection (--psm 0): Devanagari / Malayalam / other
  2. Devanagari pages only: quick Hindi OCR and a count of common Hindi vs Marathi
     words, because Hindi and Marathi share the Devanagari script
  3. Layout hints from the ground-truth labels: number of boxes to read,
     single/multi column, table, figure

Writes:
  page_candidates.csv   - one row per page
  page_candidates.html  - thumbnails of the Hindi / Malayalam candidates with checkboxes;
                          tick the pages you want, then "Save list" -> selected_pages.txt
"""

import csv
import html
import os
import re
import subprocess
import yaml
from tqdm import tqdm

TESSERACT = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
DATA_DIR = "dataset_root"
FOLDER = "page_selection"   # run from the project folder:  python page_selection/pick_pages.py
SPLITS = ["test", "val"]
MIN_SCRIPT_CONF = 5.0   # below this, Tesseract's script guess is unreliable

names = yaml.safe_load(open("data.yaml"))["names"]
roles = yaml.safe_load(open("class_roles.yaml"))["roles"]

# Very common words that appear in one language but not the other
HINDI_WORDS = ["है", "हैं", "और", "में", "नहीं", "किया", "करें", "गया"]
MARATHI_WORDS = ["आहे", "आहेत", "आणि", "च्या", "नाही", "केले", "करा", "होते"]


def detect_script(image_path):
    out = subprocess.run([TESSERACT, image_path, "-", "--psm", "0"],
                         capture_output=True, text=True, encoding="utf-8", errors="ignore")
    text = out.stdout + out.stderr
    script = re.search(r"Script:\s*(\S+)", text)
    conf = re.search(r"Script confidence:\s*([\d.]+)", text)
    return (script.group(1) if script else "unknown"), (float(conf.group(1)) if conf else 0.0)


def hindi_or_marathi(image_path):
    out = subprocess.run([TESSERACT, image_path, "-", "-l", "hin"],
                         capture_output=True, text=True, encoding="utf-8", errors="ignore")
    words = re.findall(r"[\u0900-\u097F]+", out.stdout)
    hi = sum(w in HINDI_WORDS for w in words)
    mr = sum(w in MARATHI_WORDS for w in words) + sum("ळ" in w for w in words)  # ळ is common in Marathi only
    if hi + mr < 3:
        guess = "unsure"
    elif hi >= 2 * mr:
        guess = "hindi"
    elif mr >= 2 * hi:
        guess = "marathi"
    else:
        guess = "unsure"
    return guess, hi, mr


def layout_hints(label_path):
    read_boxes, has_table, has_figure, left, right = 0, False, False, False, False
    for line in open(label_path):
        parts = line.split()
        if len(parts) != 5:
            continue
        cls = names[int(parts[0])]
        xc, yc, w, h = map(float, parts[1:])
        if roles.get(cls) in ("read", "special"):
            read_boxes += 1
        has_table |= cls == "table"
        has_figure |= cls == "figure"
        if cls == "paragraph" and w < 0.45:   # narrow paragraph = probably a column
            left |= xc < 0.4
            right |= xc > 0.6
    return read_boxes, ("multi" if left and right else "single"), has_table, has_figure


rows = []
for split in SPLITS:
    image_dir = os.path.join(DATA_DIR, "images", split)
    for image_name in tqdm(sorted(os.listdir(image_dir)), desc=f"Checking {split}"):
        image_path = os.path.join(image_dir, image_name)
        script, conf = detect_script(image_path)
        language, hi, mr = "", 0, 0
        if script == "Malayalam" and conf >= MIN_SCRIPT_CONF:
            language = "malayalam"
        elif script == "Devanagari" and conf >= MIN_SCRIPT_CONF:
            language, hi, mr = hindi_or_marathi(image_path)
        label_path = os.path.join(DATA_DIR, "labels", split, image_name.replace(".png", ".txt"))
        read_boxes, columns, has_table, has_figure = layout_hints(label_path)
        rows.append(dict(split=split, image=image_name, script=script, script_conf=conf,
                         language=language, hindi_words=hi, marathi_words=mr,
                         read_boxes=read_boxes, columns=columns, has_table=has_table, has_figure=has_figure))

with open(os.path.join(FOLDER, "page_candidates.csv"), "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=rows[0].keys())
    writer.writeheader()
    writer.writerows(rows)

# ---- Summary ----
print("\nPages per language guess:")
for lang in ["hindi", "malayalam", "marathi", "unsure"]:
    pages = [r for r in rows if r["language"] == lang]
    multi = sum(r["columns"] == "multi" for r in pages)
    print(f"  {lang:<10} {len(pages):4d}   (multi-column: {multi}, with table: {sum(r['has_table'] for r in pages)}, "
          f"with figure: {sum(r['has_figure'] for r in pages)})")

# ---- Gallery: Hindi, Malayalam and unsure Devanagari pages with at least 3 boxes to read ----
order = {"hindi": 0, "malayalam": 1, "unsure": 2}
candidates = [r for r in rows if r["language"] in order and r["read_boxes"] >= 3]
candidates.sort(key=lambda r: (order[r["language"]], r["columns"] != "multi", -r["read_boxes"]))

cards = []
for r in candidates:
    page_id = f"{r['split']}/{r['image']}"
    tags = [r["language"], r["columns"] + "-column", f"{r['read_boxes']} boxes"]
    if r["has_table"]:
        tags.append("table")
    if r["has_figure"]:
        tags.append("figure")
    if r["language"] == "unsure":
        tags.append(f"hi:{r['hindi_words']} mr:{r['marathi_words']}")
    cards.append(f"""
  <label class="card {r['language']}">
    <img loading="lazy" src="../{DATA_DIR}/images/{page_id}">
    <div><input type="checkbox" value="{html.escape(page_id)}"> {html.escape(page_id)}</div>
    <div class="tags">{' · '.join(tags)}</div>
  </label>""")

page = f"""<!doctype html>
<html><head><meta charset="utf-8"><title>Page Picker</title>
<style>
  body {{ font-family: sans-serif; margin: 16px; background: #fafafa; }}
  .bar {{ position: sticky; top: 0; background: #fff; padding: 10px; border-bottom: 1px solid #ccc; z-index: 1; }}
  .grid {{ display: grid; grid-template-columns: repeat(auto-fill, minmax(230px, 1fr)); gap: 12px; margin-top: 12px; }}
  .card {{ background: #fff; border: 2px solid #ddd; border-radius: 6px; padding: 6px; font-size: 13px; cursor: pointer; }}
  .card:has(input:checked) {{ border-color: #2a7; background: #efe; }}
  .card img {{ width: 100%; height: 300px; object-fit: contain; background: #eee; }}
  .tags {{ color: #555; margin-top: 4px; }}
  .hindi .tags {{ color: #a50; }} .malayalam .tags {{ color: #05a; }}
</style></head><body>
<div class="bar">
  <b>Pick ~60 Hindi + ~60 Malayalam pages</b> (mix single/multi-column, tables, figures; skip Marathi).
  Selected: <b id="count">0</b> &nbsp; <button onclick="save()">Save list</button>
</div>
<div class="grid">{''.join(cards)}
</div>
<script>
  const boxes = document.querySelectorAll('input[type=checkbox]');
  boxes.forEach(b => b.addEventListener('change', () => {{
    document.getElementById('count').textContent = [...boxes].filter(x => x.checked).length;
  }}));
  function save() {{
    const picked = [...boxes].filter(x => x.checked).map(x => x.value).join('\\n');
    const a = document.createElement('a');
    a.href = URL.createObjectURL(new Blob([picked + '\\n'], {{type: 'text/plain'}}));
    a.download = 'selected_pages.txt';
    a.click();
  }}
</script>
</body></html>"""
open(os.path.join(FOLDER, "page_candidates.html"), "w", encoding="utf-8").write(page)
print(f"\n{len(candidates)} candidate pages in page_selection/page_candidates.html - open it in your browser.")
