"""
Full reading pipeline: page image -> layout boxes -> reading order -> OCR -> speech.

  1. boxes:   the fine-tuned YOLOv10x layout detector (or the ground-truth boxes with --gt-boxes)
  2. filter:  headers, footers, page numbers, ads ... are skipped (class_roles.yaml);
              figures and formulas are kept in the order but not spoken
  3. order:   the pairwise MLP (P2), or XY-cut with --order xy-cut
  4. OCR:     Tesseract on each box, Hindi or Tamil + English (English text is read too);
              in an Indic box, stray Latin words are OCR noise and are dropped
  5. speech:  Meta MMS-TTS (facebook/mms-tts-hin / mms-tts-tam), runs on CPU;
              boxes that are mostly English use the English voice (mms-tts-eng)

    python read_page.py --page val/val_00140.png --language tamil
    python read_page.py --image my_scan.png --language hindi
    python read_page.py --page val/val_00140.png --language tamil --order xy-cut --no-audio

Saves to outputs/<page name>/: order.png (numbered boxes), text.txt, boxes.json, speech.wav
"""

import argparse
import io
import json
import os
import re
import subprocess

import numpy as np
from PIL import Image, ImageDraw

from reading_order_utils import DATA_DIR, NAMES, ROLES, is_read, load_gt_boxes, load_pair_mlp, mlp_order, xy_cut

TESSERACT = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
DETECTOR = "yolov10x_indicdlp.pt"
OCR_LANG = {"hindi": "hin+eng", "tamil": "tam+eng"}
TTS_MODEL = {"hindi": "facebook/mms-tts-hin", "tamil": "facebook/mms-tts-tam", "english": "facebook/mms-tts-eng"}
NOT_SPOKEN = {"figure", "formula"}   # kept in the order, but there is no text to read aloud
PAUSE_SECONDS = 0.4                  # silence between two boxes

_models = {}   # loaded once, reused (the Gradio demo calls read_page many times)


# ============================================================
# 1. Layout boxes
# ============================================================

def detect_boxes(image, conf=0.25):
    from ultralytics import YOLO
    if "detector" not in _models:
        _models["detector"] = YOLO(DETECTOR)
    result = _models["detector"].predict(image, conf=conf, verbose=False)[0]
    W, H = image.size
    boxes = []
    for (x1, y1, x2, y2), c, score in zip(result.boxes.xyxy.tolist(), result.boxes.cls.tolist(),
                                          result.boxes.conf.tolist()):
        boxes.append({"id": len(boxes), "class": NAMES[int(c)], "score": round(score, 3),
                      "bbox": [x1 / W, y1 / H, x2 / W, y2 / H]})
    return boxes


# ============================================================
# 2-3. Reading order
# ============================================================

def order_boxes(boxes, method="mlp"):
    read = [b for b in boxes if is_read(b)]
    if method == "xy-cut":
        return xy_cut(read)
    if "mlp" not in _models:
        _models["mlp"] = load_pair_mlp()
    model, xycut_feature = _models["mlp"]
    return mlp_order(model, read, xycut_feature)


# ============================================================
# 4. OCR
# ============================================================

def ocr_box(image, box, language):
    W, H = image.size
    x1, y1, x2, y2 = box["bbox"]
    pad = 4
    crop = image.crop((max(0, x1 * W - pad), max(0, y1 * H - pad), min(W, x2 * W + pad), min(H, y2 * H + pad)))
    if crop.height < 60:   # Tesseract reads small text better when it is bigger
        crop = crop.resize((crop.width * 2, crop.height * 2), Image.LANCZOS)
    buf = io.BytesIO()
    crop.save(buf, format="PNG")
    out = subprocess.run([TESSERACT, "stdin", "stdout", "-l", OCR_LANG[language], "--psm", "6"],
                         input=buf.getvalue(), capture_output=True)
    text = out.stdout.decode("utf-8", errors="ignore").replace(chr(0x200C), "")   # Tesseract adds invisible joiners in Tamil
    text = re.sub(r"\s+", " ", text).strip()   # one line per box
    if latin_share(text) < 0.5:
        # an Indic box: Latin letters here are OCR noise from the English model, drop those words
        text = " ".join(w for w in text.split() if not re.search(r"[A-Za-z]", w))
    return text


def latin_share(text):
    """Share of letters that are Latin (English) rather than Devanagari / Tamil."""
    latin = len(re.findall(r"[A-Za-z]", text))
    indic = len(re.findall(r"[\u0900-\u097F\u0B80-\u0BFF]", text))   # Devanagari, Tamil
    return latin / max(1, latin + indic)


# ============================================================
# 5. Speech
# ============================================================

def tts_model(voice):
    from transformers import AutoTokenizer, VitsModel
    key = "tts_" + voice
    if key not in _models:
        _models[key] = (AutoTokenizer.from_pretrained(TTS_MODEL[voice]), VitsModel.from_pretrained(TTS_MODEL[voice]))
    return _models[key]


def speak(texts, language):
    """One waveform for all boxes, with a short pause between boxes. Returns (audio, sample_rate).
    A box that is mostly English is spoken with the English voice (all MMS voices use 16 kHz)."""
    import torch
    pieces, rate = [], 16000
    for text in texts:
        tokenizer, model = tts_model("english" if latin_share(text) >= 0.5 else language)
        rate = model.config.sampling_rate
        # long boxes are spoken sentence by sentence (MMS works best on short inputs)
        for sentence in re.split(r"(?<=[।.!?])\s+", text):
            inputs = tokenizer(sentence, return_tensors="pt")
            if inputs.input_ids.shape[1] < 2:   # nothing the voice can say
                continue
            with torch.no_grad():
                pieces.append(model(**inputs).waveform[0].numpy())
        pieces.append(np.zeros(int(PAUSE_SECONDS * rate), dtype=np.float32))
    return (np.concatenate(pieces) if pieces else np.zeros(1, dtype=np.float32)), rate


# ============================================================
# Whole pipeline
# ============================================================

def draw_order(image, all_boxes, ordered):
    """Page with skipped boxes in grey and read boxes numbered in reading order."""
    img = image.convert("RGB")
    d = ImageDraw.Draw(img)
    W, H = img.size
    size = max(14, W // 50)
    for b in all_boxes:
        if not is_read(b):
            x1, y1, x2, y2 = b["bbox"]
            d.rectangle([x1 * W, y1 * H, x2 * W, y2 * H], outline=(160, 160, 160), width=2)
    for k, b in enumerate(ordered, 1):
        x1, y1, x2, y2 = b["bbox"]
        color = (142, 36, 170) if ROLES.get(b["class"]) == "special" else (21, 101, 192)
        d.rectangle([x1 * W, y1 * H, x2 * W, y2 * H], outline=color, width=3)
        d.rectangle([x1 * W, y1 * H, x1 * W + 1.8 * size, y1 * H + 1.3 * size], fill=(211, 47, 47))
        d.text((x1 * W + 3, y1 * H + 1), str(k), fill="white", font_size=size)
    return img


def read_page(image, language, boxes=None, order="mlp", audio=True):
    """image: PIL image. boxes: ground-truth boxes, or None to run the detector.
    Returns dict with ordered boxes (with OCR text), the numbered page image, and audio."""
    image = image.convert("RGB")
    all_boxes = boxes if boxes is not None else detect_boxes(image)
    ordered = order_boxes(all_boxes, order)
    for b in ordered:
        b["text"] = "" if b["class"] in NOT_SPOKEN else ocr_box(image, b, language)
    result = {"boxes": ordered, "all_boxes": all_boxes, "order_image": draw_order(image, all_boxes, ordered)}
    if audio:
        result["audio"], result["rate"] = speak([b["text"] for b in ordered if b["text"]], language)
    return result


if __name__ == "__main__":
    import sys
    import soundfile as sf
    sys.stdout.reconfigure(encoding="utf-8")   # Tamil / Hindi text in the terminal

    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--page", help="a dataset page like val/val_00140.png")
    group.add_argument("--image", help="any page image file")
    parser.add_argument("--language", required=True, choices=["hindi", "tamil"])
    parser.add_argument("--order", default="mlp", choices=["mlp", "xy-cut"])
    parser.add_argument("--gt-boxes", action="store_true", help="use ground-truth boxes instead of the detector")
    parser.add_argument("--no-audio", action="store_true")
    args = parser.parse_args()

    path = os.path.join(DATA_DIR, "images", args.page) if args.page else args.image
    boxes = load_gt_boxes(args.page) if args.gt_boxes else None
    if args.gt_boxes and not args.page:
        parser.error("--gt-boxes needs --page")

    result = read_page(Image.open(path), args.language, boxes, args.order, audio=not args.no_audio)

    out_dir = os.path.join("outputs", os.path.splitext(os.path.basename(path))[0] + f"_{args.order}")
    os.makedirs(out_dir, exist_ok=True)
    result["order_image"].save(os.path.join(out_dir, "order.png"))
    with open(os.path.join(out_dir, "text.txt"), "w", encoding="utf-8") as f:
        for k, b in enumerate(result["boxes"], 1):
            f.write(f"[{k}] ({b['class']}) {b['text']}\n\n")
    json.dump(result["boxes"], open(os.path.join(out_dir, "boxes.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)
    if "audio" in result:
        sf.write(os.path.join(out_dir, "speech.wav"), result["audio"], result["rate"])

    print(f"{len(result['all_boxes'])} boxes found, {len(result['boxes'])} read in this order:")
    for k, b in enumerate(result["boxes"], 1):
        print(f"  {k:>2}. {b['class']:<20} {b['text'][:70]}")
    print(f"Saved {out_dir}/ (order.png, text.txt, boxes.json{', speech.wav' if 'audio' in result else ''})")
