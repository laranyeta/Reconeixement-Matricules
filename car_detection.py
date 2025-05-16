#####################################
### CAR DETECTION (DEEP LEARNING) ###
#####################################

from ultralytics import YOLO
import torch

model = YOLO("models/yolov8n.pt")

#FASE TRAINING (nomes realitzar un cop)
'''results = model.train(
    data="car_yolo/config_car.yaml",
    epochs=15,          
    imgsz=416,          
    batch=8,            
    device="cpu",  #0 -> gpu, "cpu" -> cpu
    cache=True,         
    workers=4,          
    project="car_yolo/runs/train_fast",
    name="yolov8n_cotxes"
)
infer = YOLO("car_yolo/runs/train_fast/yolov8n_cotxes/weights/best.pt")
results = infer.predict("database/car/images/val/IMG_0973_jpg.rf.f7a6938e4d94b23a29f7c50020c6f2bd.jpg",save=True)'''