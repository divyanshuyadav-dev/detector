from ultralytics import YOLO
import torch

model = YOLO("yolo26n.pt")

result = model.train(
    
    data = 'dataset/data.yaml',
    epochs = 100,
    exist_ok = true,
    device = 0 if torch.cuda.is_available() else 'cpu',
    patience = 5
)