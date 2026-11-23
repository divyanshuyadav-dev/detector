from ultralytics import YOLO
import torch


model = YOLO('best.pt')

results = model.track(
    source='dataset/inference_media/videos/16024729_360_640_59fps.mp4',
    stream= False,
    tracker = 'bytetrack.yaml',
    save = True
)

print(results)