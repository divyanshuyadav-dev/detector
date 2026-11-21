from ultralytics import YOLO

model = YOLO("yolo26n.pt")

result = model.train(
    
    data = 'dataset/data.yaml',
    epochs = 1
)