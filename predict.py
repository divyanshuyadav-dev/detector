from ultralytics import YOLO
import torch


model = YOLO("best.pt")

results = model.predict(
    source='dataset/inference_media/',
    save=True,
    device = 0 if torch.cuda.is_available() else "cpu"
)

for result in results:

    print(f"at: {result.path}")
    for box in result.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())  # pixel coordinates
        conf = float(box.conf[0])                        # confidence score
        cls_id = int(box.cls[0])                         # class index
        label = result.names[cls_id]                     # class name

        print(f"{label}: {conf:.2f} at [{x1}, {y1}, {x2}, {y2}]")
