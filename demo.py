"""
Demo: read a Hindi or Tamil page aloud in the right order.

    python demo.py        then open http://127.0.0.1:7860

Tab "Read a page":     upload a page (or click an example), see the numbered reading order,
                       the OCR text per box, and listen to it.
Tab "Listening test":  for a listener (e.g. a Tamil speaker). Each page has two versions, A and B,
                       read in a different order (one is the MLP, one is XY-cut; the listener is not told).
                       The listener picks the better order and rates how clear the voice is.
                       Answers are saved to results/ratings.csv. Clips come from make_rating_clips.py.
"""

import csv
import os
import time

import gradio as gr
from PIL import Image

from read_page import read_page, speak
from reading_order_utils import DATA_DIR, load_split_pages

CLIP_DIR = os.path.join("outputs", "rating_clips")
RATINGS = os.path.join("results", "ratings.csv")

# ============================================================
# Tab 1: read a page
# ============================================================
examples = []
for p in load_split_pages()["test"]:
    if p["columns"] == "multi" or len(examples) < 4:
        examples.append([os.path.join(DATA_DIR, "images", p["page"]), p["language"]])


def run(image, language, order, max_boxes):
    if image is None:
        raise gr.Error("Upload a page image first.")
    start = time.time()
    result = read_page(image, language, order=order, audio=False)
    rows = [[k, b["class"], b["text"]] for k, b in enumerate(result["boxes"], 1)]
    spoken = [b["text"] for b in result["boxes"][:int(max_boxes)] if b["text"]]
    audio, rate = speak(spoken, language)
    info = (f"{len(result['all_boxes'])} boxes found, {len(result['boxes'])} read "
            f"(headers, footers, page numbers skipped). Spoke the first {min(int(max_boxes), len(rows))} "
            f"in {time.time() - start:.0f} s.")
    return result["order_image"], rows, (rate, audio), info


# ============================================================
# Tab 2: listening test
# ============================================================

def load_pages():
    """One entry per page: its image and the two clips (A, B)."""
    path = os.path.join(CLIP_DIR, "clips.csv")
    if not os.path.exists(path):
        return []
    pages = {}
    for c in csv.DictReader(open(path, encoding="utf-8")):
        pages.setdefault(c["page"], {"page": c["page"], "language": c["language"], "image": c["image"]})[c["version"]] = c
    return list(pages.values())


rating_pages = load_pages()


def show(i):
    if not rating_pages:   # clips may have been made after the demo started
        rating_pages.extend(load_pages())
    if not rating_pages:
        return None, None, None, "No clips yet: run  python make_rating_clips.py"
    if i >= len(rating_pages):
        return None, None, None, "All pages done. Thank you!"
    p = rating_pages[i]
    return (Image.open(os.path.join(CLIP_DIR, p["image"])), os.path.join(CLIP_DIR, p["A"]["clip"] + ".wav"),
            os.path.join(CLIP_DIR, p["B"]["clip"] + ".wav"),
            f"Page {i + 1} of {len(rating_pages)} ({p['language']}). Look at the page, play A and B, then answer.")


def save(i, rater, better, clear_a, clear_b, comment):
    if not rater.strip():
        raise gr.Error("Please type your name first.")
    if not better:
        raise gr.Error("Please choose which version reads the page in the better order.")
    p = rating_pages[i]
    row = dict(time=time.strftime("%Y-%m-%d %H:%M:%S"), rater=rater.strip(), page=p["page"], language=p["language"],
               better=better, clarity_A=clear_a, clarity_B=clear_b,
               method_A=p["A"]["method"], method_B=p["B"]["method"], comment=comment)
    new_file = not os.path.exists(RATINGS)
    with open(RATINGS, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(row.keys()))
        if new_file:
            w.writeheader()
        w.writerow(row)
    i += 1
    return (i, *show(i), None, 3, 3, "")


with gr.Blocks(title="Indic page reader") as demo:
    gr.Markdown("# Indic page reader\nLayout detection → reading order → OCR → speech, for Hindi and Tamil pages.")

    with gr.Tab("Read a page"):
        with gr.Row():
            with gr.Column():
                page_in = gr.Image(type="pil", label="Page image")
                language = gr.Radio(["hindi", "tamil"], value="hindi", label="Language")
                order = gr.Radio([("Our model (pairwise MLP)", "mlp"), ("XY-cut (rule)", "xy-cut")], value="mlp",
                                 label="Reading order")
                max_boxes = gr.Slider(1, 40, value=6, step=1, label="Speak the first N boxes (speech is slow on CPU)")
                go = gr.Button("Read the page", variant="primary")
                gr.Examples(examples, [page_in, language], label="Test pages")
            with gr.Column():
                info = gr.Markdown()
                page_out = gr.Image(label="Reading order (red numbers; grey boxes are skipped)")
                audio_out = gr.Audio(label="Speech")
                text_out = gr.Dataframe(headers=["#", "type", "OCR text"], wrap=True, label="Text in reading order")
        go.click(run, [page_in, language, order, max_boxes], [page_out, text_out, audio_out, info])

    with gr.Tab("Listening test"):
        gr.Markdown("You will hear the start of each page read aloud twice, **A** and **B**, in a different order. "
                    "Follow along on the page. Pick the version whose order makes more sense, "
                    "and rate how clear each voice is (1 = can't understand, 5 = very clear).")
        index = gr.State(0)
        rater = gr.Textbox(label="Your name")
        status = gr.Markdown()
        with gr.Row():
            page_img = gr.Image(label="Page", height=700)
            with gr.Column():
                clip_a = gr.Audio(label="Version A")
                clear_a = gr.Slider(1, 5, value=3, step=1, label="How clear is A?")
                clip_b = gr.Audio(label="Version B")
                clear_b = gr.Slider(1, 5, value=3, step=1, label="How clear is B?")
                better = gr.Radio(["A", "B", "both fine", "both wrong"], label="Which reads the page in the better order?")
                comment = gr.Textbox(label="Comment (optional)")
                next_btn = gr.Button("Save & next", variant="primary")
        demo.load(show, [index], [page_img, clip_a, clip_b, status])
        next_btn.click(save, [index, rater, better, clear_a, clear_b, comment],
                       [index, page_img, clip_a, clip_b, status, better, clear_a, clear_b, comment])

if __name__ == "__main__":
    demo.launch(allowed_paths=[os.path.abspath(CLIP_DIR)])   # let Gradio serve the listening-test clips
