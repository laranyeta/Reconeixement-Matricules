############### RECONEIXEMENT DE MATRÍCULES DE VEHICLES #####################
# Autors: Pau Bofill, Lara Castillejo, Júlia Lipin Gener
# Curs: 2024-2025

from car_detection import *
from plate_detection import *
from plate_ocr_recog import *
from utils import *
import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

# MOSTRAR IMATGE AGAFADA DE TEST
img = cv2.imread("test/test4.jpg")
plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
plt.title("Imatge de test")
plt.axis('off')
plt.show()

# DETECCIO DE COTXES
model_cotxe = YOLO("car_yolo/runs/train_fast/yolov8n_cotxes2/weights/best.pt")
result_cotxe = model_cotxe.predict("test/test4.jpg", save=True, imgsz=416)

detected_car = cv2.imread("runs/detect/predict66/test4.jpg")
cropped_car = crop_car(result_cotxe, detected_car)
cv2.imwrite("output/cropped_car.jpg", cropped_car)

plt.imshow(cv2.cvtColor(cropped_car, cv2.COLOR_BGR2RGB))
plt.title("Regió Bounding-Box del cotxe detectat")
plt.axis('off')
plt.show()

# LOCALITZACIO DE MATRICULA
plate_bounding_box = detect_plate(cropped_car)
image_with_box = display_bounding_box(cropped_car, plate_bounding_box)
plt.imshow(cv2.cvtColor(image_with_box, cv2.COLOR_BGR2RGB))
plt.title("Bounding Box de la matrícula")
plt.axis('off')
plt.show()

cropped_plate = crop_plate(plate_bounding_box, cropped_car)
cv2.imwrite("output/cropped_plate.jpg", cropped_plate)

plt.imshow(cv2.cvtColor(cropped_plate, cv2.COLOR_BGR2RGB))
plt.title("Regió Bounding-Box de la matrícula detectada")
plt.axis('off')
plt.show()

# OCR MATRÍCULA
plate_img = cv2.imread("output/cropped_plate.jpg")
result = predict_plate_text(plate_img)
print("Matricula detectada:", result)


