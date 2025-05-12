############### RECONEIXEMENT DE MATRÍCULES DE VEHICLES #####################
# RECONEIXEMENT DE CARÀCTERS DE LA MATRÍCULA (OCR)

from ultralytics import YOLO
from utils import *
import torch

# FASE TRAINING DEL MODEL
model = YOLO("ocr_yolo/runs/train_fast/yolov8n_ocr/weights/best.pt")

results = model.train(
    data="ocr_yolo/config_ocr.yaml",
    epochs=15,          
    imgsz=416,          
    batch=8,            
    device="cpu",  #0 -> gpu, "cpu" -> cpu
    cache=True,         
    workers=4,          
    project="ocr_yolo/runs/train_fast",
    name="yolov8n_ocr"
)
infer = YOLO("ocr_yolo/runs/train_fast/yolov8n_ocr/weights/best.pt")
results = model.val()

#FASE TEST DEL MODEL
#results = model.predict("img_prova.jpeg", save=True, imgsz=416)

#plate_text = plate_to_text(model, results)
#print("Matricula detectada: ", plate_text)