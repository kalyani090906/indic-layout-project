"""
Reading-order labeling tool. Runs a small local web server; open http://localhost:8000 in your browser.

    python label_tool.py              # first labeling pass, all 120 pages
    python label_tool.py --relabel    # second pass on the 15 relabel pages (about a week later)

Each page shows its ground-truth boxes, pre-numbered in XY-cut order. Fix the order, then Save.
One JSON file per page is written to reading_order/labels/ (or reading_order/labels_relabel/).
"""

import argparse
import csv
import json
import os
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

from PIL import Image
from reading_order_utils import DATA_DIR, ROLES, is_read, load_gt_boxes, xy_cut

parser = argparse.ArgumentParser()
parser.add_argument("--relabel", action="store_true", help="second pass on the relabel pages")
parser.add_argument("--review", action="store_true", help="only the pages in reading_order/review_pages.csv")
parser.add_argument("--port", type=int, default=8000)
args = parser.parse_args()

PAGES = list(csv.DictReader(open("reading_order/pages.csv", encoding="utf-8")))
if args.relabel:
    PAGES = [p for p in PAGES if p["relabel"] == "1"]
if args.review:
    reasons = {r["page"]: r["reason"] for r in csv.DictReader(open("reading_order/review_pages.csv", encoding="utf-8"))}
    PAGES = [dict(p, reason=reasons[p["page"]]) for p in PAGES if p["page"] in reasons]
OUT_DIR = os.path.join("reading_order", "labels_relabel" if args.relabel else "labels")
os.makedirs(OUT_DIR, exist_ok=True)


def label_path(page_id):
    return os.path.join(OUT_DIR, page_id.replace("/", "_").replace(".png", ".json"))


def is_done(page_id):
    """Labeled. In --review mode: saved again after the review list was made."""
    if not os.path.exists(label_path(page_id)):
        return False
    if args.review:
        return os.path.getmtime(label_path(page_id)) > os.path.getmtime("reading_order/review_pages.csv")
    return True


def page_data(page_id):
    """Boxes with the saved order if the page was labeled before, else the XY-cut pre-fill."""
    if os.path.exists(label_path(page_id)):
        saved = json.load(open(label_path(page_id), encoding="utf-8"))
        return saved["boxes"], True
    boxes = load_gt_boxes(page_id)
    order = xy_cut([b for b in boxes if is_read(b)])
    rank = {b["id"]: i + 1 for i, b in enumerate(order)}
    for b in boxes:
        b["order"] = rank.get(b["id"])          # None = skipped
        b["role"] = ROLES.get(b["class"], "read")
    return boxes, False


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def send(self, body, ctype="application/json", code=200):
        data = body if isinstance(body, bytes) else body.encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        url = urlparse(self.path)
        query = parse_qs(url.query)
        if url.path == "/":
            self.send(open("label_tool.html", encoding="utf-8").read(), "text/html; charset=utf-8")
        elif url.path == "/api/pages":
            self.send(json.dumps([dict(p, done=is_done(p["page"])) for p in PAGES]))
        elif url.path == "/api/page":
            page_id = query["id"][0]
            boxes, done = page_data(page_id)
            w, h = Image.open(os.path.join(DATA_DIR, "images", page_id)).size
            self.send(json.dumps({"page": page_id, "width": w, "height": h, "boxes": boxes, "done": done,
                                  "relabel": args.relabel}))
        elif url.path == "/image":
            page_id = query["id"][0]
            self.send(open(os.path.join(DATA_DIR, "images", page_id), "rb").read(), "image/png")
        else:
            self.send("not found", "text/plain", 404)

    def do_POST(self):
        if urlparse(self.path).path != "/api/save":
            return self.send("not found", "text/plain", 404)
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        meta = next(p for p in PAGES if p["page"] == body["page"])
        orders = sorted(b["order"] for b in body["boxes"] if b["order"] is not None)
        if orders != list(range(1, len(orders) + 1)):
            return self.send(json.dumps({"ok": False, "error": "order numbers must be 1..N with no gaps"}), code=400)
        record = {"page": body["page"], "language": meta["language"], "split": meta["split"],
                  "boxes": body["boxes"], "seconds_spent": body.get("seconds"),
                  "saved_at": time.strftime("%Y-%m-%d %H:%M:%S")}
        json.dump(record, open(label_path(body["page"]), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
        self.send(json.dumps({"ok": True}))


print(f"{'Relabel' if args.relabel else 'Labeling'} {len(PAGES)} pages -> {OUT_DIR}/")
print(f"Open http://localhost:{args.port} in your browser. Press Ctrl+C here to stop.")
ThreadingHTTPServer(("127.0.0.1", args.port), Handler).serve_forever()
