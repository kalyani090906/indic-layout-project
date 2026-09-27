from ultralytics import YOLO

model = YOLO("doclayout_yolo_indicdlp.pt")
metrics = model.val(data="data.yaml", split="test")