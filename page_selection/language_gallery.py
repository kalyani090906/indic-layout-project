"""
Gallery for tagging page languages by eye (Tesseract's script detection was unreliable on these pages).

For every val + test page it saves a close-up of the page's largest text block (from the
ground-truth labels), so the script is easy to recognise:
  Hindi     - Devanagari: letters hang from a top line (हिंदी); Marathi looks the same, but uses आहे / आणि
  Tamil     - angular letters, no top line (தமிழ்); Telugu/Kannada are curlier with little "hats"

Open language_gallery.html, click Hindi / Tamil on the pages you recognise (ignore the rest),
then "Save tags" -> language_tags.csv. Put that file in the project folder.
"""

import csv
import os
import yaml
from PIL import Image
from tqdm import tqdm

DATA_DIR = "dataset_root"
FOLDER = "page_selection"   # run from the project folder:  python page_selection/language_gallery.py
THUMB_DIR = os.path.join(FOLDER, "language_thumbs")
TEXT_CLASSES = {"paragraph", "headline", "section-title", "ordered-list", "unordered-list",
                "first-level-question", "options", "sidebar"}

names = yaml.safe_load(open("data.yaml"))["names"]
os.makedirs(THUMB_DIR, exist_ok=True)

# Hints from pick_pages.py, if it was run (Devanagari pages and the Hindi/Marathi word count)
hints = {}
if os.path.exists(os.path.join(FOLDER, "page_candidates.csv")):
    for r in csv.DictReader(open(os.path.join(FOLDER, "page_candidates.csv"), encoding="utf-8")):
        hints[f"{r['split']}/{r['image']}"] = r

# Median Tamil-OCR word confidence per page (real Tamil pages score 60-96, others mostly below 50)
tamil = {}
if os.path.exists(os.path.join(FOLDER, "tamil_scores.csv")):
    for r in csv.DictReader(open(os.path.join(FOLDER, "tamil_scores.csv"), encoding="utf-8")):
        tamil[r["page"]] = float(r["tamil_conf"])

pages = []
for split in ["test", "val"]:
    for image_name in tqdm(sorted(os.listdir(os.path.join(DATA_DIR, "images", split))), desc=f"Close-ups {split}"):
        page_id = f"{split}/{image_name}"
        page = Image.open(os.path.join(DATA_DIR, "images", split, image_name)).convert("RGB")
        W, H = page.size

        # largest text box on the page; fall back to the middle of the page
        best, best_area = None, 0
        for line in open(os.path.join(DATA_DIR, "labels", split, image_name.replace(".png", ".txt"))):
            p = line.split()
            if len(p) != 5 or names[int(p[0])] not in TEXT_CLASSES:
                continue
            xc, yc, w, h = map(float, p[1:])
            if 0 < w <= 1 and 0 < h <= 1 and w * h > best_area:
                best, best_area = (xc, yc, w, h), w * h
        xc, yc, w, h = best if best else (0.5, 0.5, 0.8, 0.3)
        h = min(h, w * 0.6)   # keep the close-up wide and short, so letters stay big
        box = (max(0, (xc - w / 2) * W), max(0, (yc - h / 2) * H), min(W, (xc + w / 2) * W), min(H, (yc + h / 2) * H))
        box = tuple(int(v) for v in box)
        if box[2] - box[0] < 10 or box[3] - box[1] < 10:   # tiny or broken box: show the whole page
            box = (0, 0, W, H)
        crop = page.crop(box)
        crop.thumbnail((420, 260))
        thumb = f"{THUMB_DIR}/{split}_{image_name.replace('.png', '.jpg')}"
        crop.save(thumb, quality=85)

        hint = hints.get(page_id, {})
        guess = hint.get("language") or ""
        tamil_conf = tamil.get(page_id, 0.0)
        if not guess and tamil_conf >= 50:
            guess = f"tamil? ({tamil_conf:.0f})"
        pages.append((page_id, os.path.relpath(thumb, FOLDER).replace(os.sep, "/"), guess, hint.get("script", ""), tamil_conf))


def sort_key(p):
    page_id, _, guess, script, tamil_conf = p
    if guess == "hindi":
        return (0, 0, page_id)
    if tamil_conf >= 60:            # Tamil OCR reads it confidently: almost surely Tamil
        return (1, -tamil_conf, page_id)
    if guess in ("unsure", "marathi"):
        return (2, 0, page_id)
    if tamil_conf >= 50:
        return (3, -tamil_conf, page_id)
    return (4, 0, script + page_id)


# Hindi guesses, then likely Tamil, then other Devanagari, then everything else
pages.sort(key=sort_key)

cards = "".join(f"""
<div class="card" data-id="{pid}">
  <a href="../{DATA_DIR}/images/{pid}" target="_blank"><img loading="lazy" src="{thumb}"></a>
  <div class="row"><span>{pid}</span><span class="hint">{guess or script}</span></div>
  <div class="row"><button data-lang="hindi">Hindi</button><button data-lang="tamil">Tamil</button></div>
</div>""" for pid, thumb, guess, script, _ in pages)

html = f"""<!doctype html>
<html><head><meta charset="utf-8"><title>Language Tags</title>
<style>
  body {{ font-family: sans-serif; margin: 16px; background: #fafafa; }}
  .bar {{ position: sticky; top: 0; background: #fff; padding: 10px; border-bottom: 1px solid #ccc; z-index: 1; }}
  .grid {{ display: grid; grid-template-columns: repeat(auto-fill, minmax(300px, 1fr)); gap: 10px; margin-top: 12px; }}
  .card {{ background: #fff; border: 3px solid #ddd; border-radius: 6px; padding: 6px; font-size: 12px; }}
  .card img {{ width: 100%; height: 170px; object-fit: contain; background: #f3f3f3; }}
  .row {{ display: flex; justify-content: space-between; gap: 6px; margin-top: 4px; }}
  .row button {{ flex: 1; padding: 6px; cursor: pointer; border: 1px solid #aaa; background: #f6f6f6; border-radius: 4px; }}
  .hint {{ color: #888; }}
  .card.hindi {{ border-color: #d80; }} .card.hindi button[data-lang=hindi] {{ background: #d80; color: #fff; }}
  .card.tamil {{ border-color: #07c; }} .card.tamil button[data-lang=tamil] {{ background: #07c; color: #fff; }}
</style></head><body>
<div class="bar">
  Click <b>Hindi</b> or <b>Tamil</b> on pages you recognise (click again to undo). Ignore all other languages.
  Marathi looks like Hindi: skip pages with आहे / आणि. Click a close-up to see the full page.<br>
  Hindi: <b id="nh">0</b> &nbsp; Tamil: <b id="nm">0</b> &nbsp; <button onclick="save()">Save tags</button>
</div>
<div class="grid">{cards}
</div>
<script>
  const KEY = 'language_tags_hi_ta';
  let tags = {{}};
  try {{ tags = JSON.parse(localStorage.getItem(KEY)) || {{}}; }} catch (e) {{}}
  function paint() {{
    document.querySelectorAll('.card').forEach(c => {{ c.classList.remove('hindi', 'tamil'); if (tags[c.dataset.id]) c.classList.add(tags[c.dataset.id]); }});
    const v = Object.values(tags);
    document.getElementById('nh').textContent = v.filter(x => x === 'hindi').length;
    document.getElementById('nm').textContent = v.filter(x => x === 'tamil').length;
    try {{ localStorage.setItem(KEY, JSON.stringify(tags)); }} catch (e) {{}}
  }}
  document.querySelectorAll('.card button').forEach(b => b.addEventListener('click', () => {{
    const id = b.closest('.card').dataset.id;
    tags[id] === b.dataset.lang ? delete tags[id] : tags[id] = b.dataset.lang;
    paint();
  }}));
  function save() {{
    const lines = ['page,language', ...Object.entries(tags).map(([p, l]) => p + ',' + l)].join('\\n');
    const a = document.createElement('a');
    a.href = URL.createObjectURL(new Blob([lines + '\\n'], {{type: 'text/csv'}}));
    a.download = 'language_tags.csv';
    a.click();
  }}
  paint();
</script>
</body></html>"""
open(os.path.join(FOLDER, "language_gallery.html"), "w", encoding="utf-8").write(html)
print(f"Saved {len(pages)} close-ups to {THUMB_DIR}/ - open language_gallery.html in your browser.")
