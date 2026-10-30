# Document Understanding for Indic Scripts: Layout-Aware Reading Order

Reads a Hindi or Tamil document page aloud in the right order:
layout boxes (YOLO) → reading order (pairwise MLP) → OCR (Tesseract) → speech (MMS-TTS).
Built on the [ai4bharat/indicdlp](https://huggingface.co/datasets/ai4bharat/indicdlp) dataset.

Run every script from this folder (paths are relative to it).

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate   # Windows
pip install ultralytics datasets tqdm pyyaml scikit-learn transformers soundfile gradio
```
Tesseract with `hin` and `tam` must be installed in `C:\Program Files\Tesseract-OCR`.

## Project map, in the order the work was done

### 1. Data
| File | What it does |
|---|---|
| `build_subset.py` | 4000 / 500 / 500 pages from the official IndicDLP train / validation / test splits → `dataset_root/` |
| `fix_labels.py` | repairs 18 label files that were normalized twice (originals in `dataset_root/labels_original/`) |
| `data.yaml`, `indic.yaml` | YOLO dataset configs (42 classes) |
| `class_roles.yaml` | each class: read, skip (header, page number …) or special (figure, table …) |

### 2. Region classifier: CNN vs transfer learning, overfitting study
| File | What it does |
|---|---|
| `make_crops.py` | crops every ground-truth box → `crops/` (25 classes) |
| `train_classifier.py` | small CNN from scratch, ResNet-50 (linear / layer4 / layer3 / layer2 / full), class weights, dropout, BatchNorm, no-pool, L1, L2, augmentation |
| `compare_runs.py` | merges all runs → `results/all_runs.csv`, `transfer_curves.png`, `regularization_curves.png`, `sparsity.csv` |
| `runs_cls/` | runs made on the laptop (C0, G6) |
| `runs_cls_downloaded/` | runs made on Colab (`colab1`, `colab2`) and Kaggle (`kaggle`) |

### 3. Layout detector study (Kaggle GPU)
| File | What it does |
|---|---|
| `train_detector.py` | YOLO transfer study (scratch / COCO / DocLayNet, full / frozen backbone) and augmentation study |
| `notebooks/kaggle_detector.ipynb` | runs it on Kaggle; results come back as `runs_det_part1.zip`, `runs_det_part2.zip` |
| `kaggle_detector_upload/` | the files uploaded to Kaggle as a dataset |
| `yolov10x_indicdlp.pt` | released IndicDLP detector (D0), also used by the reading pipeline |
| `yolov8n-doclaynet.pt` | YOLOv8n pre-trained on DocLayNet (start point for D4 / D5) |

### 4. Reading-order dataset (Hindi + Tamil, 120 pages labeled by hand)
| File | What it does |
|---|---|
| `page_selection/` | how the pages were chosen: layout hints (`pick_pages.py`), language tagging by eye (`language_gallery.py` → `language_tags.csv`), final choice and split (`select_pages.py` → `reading_order/pages.csv`) |
| `label_tool.py` + `label_tool.html` | browser tool for labeling reading order (`--review`, `--relabel`) |
| `review_pages.py` | lists the pages to re-check after the nested-box fix |
| `reading_order/` | `pages.csv` and the labels (`labels/`, one JSON per page) |

### 5. Reading-order models
| File | What it does |
|---|---|
| `reading_order_utils.py` | shared code: loading boxes and labels, raster and XY-cut baselines, metrics (Kendall tau, pairwise accuracy), pairwise MLP helpers |
| `eval_baselines.py` | raster vs XY-cut → `results/baselines.csv` |
| `train_pairwise.py` | pairwise MLP (P1, P2) |
| `train_transformer.py` | transformer, from scratch or pre-trained on XY-cut pseudo-labels (R1–R4) |
| `runs_order/` | all ordering runs + `summary.csv`; `P2_mlp_xycut/model.pt` is the model used by the pipeline |

### 6. Full pipeline, demo, listening test
| File | What it does |
|---|---|
| `read_page.py` | page image → boxes → order → OCR → speech (`outputs/<page>/`) |
| `eval_end_to_end.py` | reading order on detected boxes instead of ground truth → `results/end_to_end.csv` |
| `make_rating_clips.py` | A/B audio clips (MLP vs XY-cut) → `outputs/rating_clips/` |
| `demo.py` | Gradio demo: read a page + listening test (answers in `results/ratings.csv`) |

### Other folders
| Folder | What it is |
|---|---|
| `notebooks/` | Colab and Kaggle notebooks (`colab_resnet`, `kaggle_resnet`, `kaggle_detector`) |
| `results/` | merged tables and plots for the report |
| `archive/` | earlier work that is no longer used: first evaluation / prediction scripts, the DocLayout-YOLO checkpoint, old detector runs, and the first dataset subset (`dataset_root_old_leaked`, built only from the train split, so its test pages leaked) |
