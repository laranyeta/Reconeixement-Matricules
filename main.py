############### RECONEIXEMENT DE MATRÍCULES DE VEHICLES #####################
# Autors: Pau Bofill, Lara Castillejo, Júlia Lipin Gener
# Curs: 2024-2025

# from ultralytics import YOLO
from car_utils import *


if __name__ == "__main__":

    model = YOLO("yolov8s.pt")

    model.train(
        data = "cardata.yaml",
        epochs=50,
        imgsz=640,
        batch=16,
        name="car_detection",
        project="runs/",
    )

    test = read_images("database/car_img/test")

    for img, filename in test:
        detections = detect_car([img], model)
        cv2.imwrite("detections/car_{filename}")



