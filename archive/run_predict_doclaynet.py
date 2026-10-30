from ultralytics import YOLO
import glob

model = YOLO("yolov8n-doclaynet.pt")

# Pick a handful of test images to run predictions on
test_images = sorted(glob.glob("dataset_root/images/test/*.png"))[:10]

results = model.predict(
    source=test_images,
    save=True,
    conf=0.25
)

print(f"Ran predictions on {len(test_images)} images.")
print("Detected classes per image:")
for r in results:
    class_ids = r.boxes.cls.tolist()
    class_names = [model.names[int(c)] for c in class_ids]
    print(f"{r.path}: {class_names}")