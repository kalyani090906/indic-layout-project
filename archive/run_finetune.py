from pathlib import Path
from ultralytics import YOLO
 
# ---- config -----------------------------------------------------------
WEIGHTS = "yolov10x_indicdlp.pt"   # the checkpoint already on disk in this project
DATA_YAML = "data.yaml"
EPOCHS = 5          # small first run to prove the pipeline works end-to-end
IMG_SIZE = 640       # lower than full 1654x2339 to keep CPU training feasible
BATCH = 4            # small batch size, CPU-friendly
PROJECT = "runs/finetune"
RUN_NAME = "yolov10x_indicdlp_ft"
# ------------------------------------------------------------------------
 
def main():
    assert Path(DATA_YAML).exists(), f"{DATA_YAML} not found in {Path.cwd()}"
    assert Path(WEIGHTS).exists(), (
        f"{WEIGHTS} not found in {Path.cwd()} — check the exact filename with "
        "Get-ChildItem -Recurse -Filter \"*.pt\" and update WEIGHTS above."
    )
 
    model = YOLO(WEIGHTS)
 
    results = model.train(
        data=DATA_YAML,
        epochs=EPOCHS,
        imgsz=IMG_SIZE,
        batch=BATCH,
        device="cpu",          # no CUDA available on this machine
        project=PROJECT,
        name=RUN_NAME,
        exist_ok=True,          # overwrite/reuse the same run folder on reruns
        patience=0,              # don't early-stop on such a short run
        plots=True,              # save curves + confusion matrix like eval runs
    )
 
    print("\nTraining finished.")
    print(f"Weights saved under: {PROJECT}/{RUN_NAME}/weights/best.pt")
    print("Next: run this fine-tuned checkpoint back through run_eval.py "
          "(point it at the new best.pt) to compare its per-class mAP against "
          "the pretrained-only baseline from Section 7.")
 
 
if __name__ == "__main__":
    main()