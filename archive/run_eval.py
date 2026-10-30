from ultralytics import YOLO

model = YOLO("yolov10x_indicdlp.pt")
metrics = model.val(data="data.yaml", split="test", name="D0_released_test")