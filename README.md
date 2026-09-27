# Indic Layout Project

Document layout detection for Indic-language documents, built on the
[ai4bharat/indicdlp](https://huggingface.co/datasets/ai4bharat/indicdlp) dataset and
evaluated across several YOLO-family detectors (YOLOv10x, YOLOv8n, DocLayout-YOLO).

## Layout

- `build_subset.py` — streams a subset of `ai4bharat/indicdlp` from Hugging Face,
  splits it 80/10/10 into train/val/test, and writes images + YOLO-format labels
  into `dataset_root/`.
- `inspect_data.py` — samples the raw dataset and prints class distribution stats.
- `data.yaml` / `indic.yaml` — YOLO dataset configs (class names + split paths).
- `run_eval.py` — evaluates `yolov10x_indicdlp.pt` on the test split.
- `run_eval_doclayout.py` — evaluates `doclayout_yolo_indicdlp.pt` on the test split.
- `run_predict_yolov10x.py` / `run_predict_doclaynet.py` — run inference with a
  given model on a handful of test images and save annotated predictions.

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate   # Windows
pip install ultralytics datasets tqdm pyyaml
```

## Building the dataset

```bash
python build_subset.py
```

This populates `dataset_root/images/{train,val,test}` and
`dataset_root/labels/{train,val,test}` (not checked into git — regenerate locally).

## Model weights

`*.pt` weight files are not checked into git (too large / regenerable). Place trained
or downloaded checkpoints in the project root, matching the filenames referenced by
the `run_*.py` scripts:

- `yolov10x_indicdlp.pt`
- `doclayout_yolo_indicdlp.pt`
- `yolov8n-doclaynet.pt`

## Running evaluation / inference

```bash
python run_eval.py
python run_eval_doclayout.py
python run_predict_yolov10x.py
python run_predict_doclaynet.py
```

Outputs (metrics, annotated images) are written to `runs/` (gitignored).
