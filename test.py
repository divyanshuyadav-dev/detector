from ultralytics import YOLO
import torch

model = YOLO("/path/to/model")

metrics = model.val(
    data = 'dataset/data.yaml',
    device = 0 if torch.cuda.is_available() else "cpu"
)

# extract data from metrics