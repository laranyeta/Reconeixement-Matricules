############### RECONEIXEMENT DE MATRÍCULES DE VEHICLES #####################
# Autors: Pau Bofill, Lara Castillejo, Júlia Lipin Gener
# Curs: 2024-2025

from ultralytics import YOLO
import torch

def main():
    model = YOLO("yolov8n.pt")

    results = model.train(
    data="config.yaml",
    epochs=15,          # Entrenament curt però útil
    imgsz=416,          # Imatges més petites per fer-ho més ràpid
    batch=8,            # Ajusta si tens més VRAM
    device=0,           # GPU
    cache=True,         # Carrega imatges a RAM
    workers=4,          # Accelerar la càrrega de dades
    project="runs/train_fast",
    name="yolov8n_cotxes"
)
    infer = YOLO("runs/train_fast/yolov8n_cotxes/weights/best.pt")
    results = infer.predict("database/car_img/images/val/IMG_0973_jpg.rf.f7a6938e4d94b23a29f7c50020c6f2bd.jpg",save=True)
if __name__ == "__main__":
    main()



