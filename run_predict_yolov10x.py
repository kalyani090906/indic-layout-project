from ultralytics import YOLO
import glob

model = YOLO("yolov10x_indicdlp.pt")

# Same 10 test images as before, for a fair side-by-side comparison
test_images = sorted(glob.glob("dataset_root/images/test/*.png"))[:10]

results = model.predict(
    source=test_images,
    save=True,
    conf=0.25
)

print(f"Ran predictions on {len(test_images)} images.")