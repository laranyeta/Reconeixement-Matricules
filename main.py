############### RECONEIXEMENT DE MATRÍCULES DE VEHICLES #####################
# Autors: Pau Bofill, Lara Castillejo, Júlia Lipin Gener
# Curs: 2024-2025
''' Aquest script s'encarrega d'executar tot el procés
    de detecció i reconeixement de matrícules en els
    tests proposats 
'''

from car_detection import *
from plate_detection import *
from plate_ocr_recog import *
from utils import *
import cv2
import matplotlib.pyplot as plt

# MOSTRAR IMATGE AGAFADA DE TEST
img = cv2.imread("test/test2.jpg") #canviar per escollir imatge de test
plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
plt.title("Cotxe escollit per a la detecció")
plt.axis('off')
plt.show()

# DETECCIO DE COTXES
model_cotxe = YOLO("car_yolo/runs/train_fast/yolov8n_cotxes2/weights/best.pt")
result_cotxe = model_cotxe.predict("test/test2.jpg", save=True, imgsz=416) #canviar per escollir imatge de test

detected_car = cv2.imread("runs/detect/predict/test2.jpg") #canviar per escollir imatge de test (tmb canviar directori predict o esborrar-los de runs/detect)
cropped_car = crop_car(result_cotxe, detected_car)
cv2.imwrite("output/img/cropped_car.jpg", cropped_car)

plt.imshow(cv2.cvtColor(detected_car, cv2.COLOR_BGR2RGB))
plt.title("Cotxe detectat a l'escena")
plt.axis('off')
plt.show()

# LOCALITZACIO DE MATRICULA
plate_bounding_box = detect_plate(cropped_car)
image_with_box = display_bounding_box(cropped_car, plate_bounding_box)
plt.imshow(cv2.cvtColor(image_with_box, cv2.COLOR_BGR2RGB))
plt.title("Matricula detectada a l'escena")
plt.axis('off')
plt.show()

cropped_plate = crop_plate(plate_bounding_box, cropped_car)
cv2.imwrite("output/img/cropped_plate.jpg", cropped_plate)

'''plt.imshow(cv2.cvtColor(cropped_plate, cv2.COLOR_BGR2RGB))
plt.title("Regió Bounding-Box de la matrícula detectada")
plt.axis('off')
plt.show()'''

only_plate = crop_nationality(cropped_plate) #retallem pais
cv2.imwrite("output/img/only_plate.jpg", only_plate)
'''plt.imshow(cv2.cvtColor(only_plate, cv2.COLOR_BGR2RGB))
plt.title("Matrícula sense pais")
plt.axis('off')
plt.show()'''

# OCR MATRÍCULA
plate_img = cv2.imread("output/img/only_plate.jpg")
result, char_imgs, char_preds = predict_plate_text(plate_img)

plt.figure(figsize=(12, 2))
for i, (char_img, pred_char) in enumerate(zip(char_imgs, char_preds)):
    plt.subplot(1, len(char_imgs), i + 1)
    plt.imshow(char_img, cmap='gray')
    plt.axis('off')
    plt.title(pred_char)
plt.suptitle("Caràcters segmentats i reconeguts")
plt.tight_layout()
plt.show()

#PLOT FINAL
output_img = img.copy()
cv2.putText(
    output_img,
    f"Matricula detectada: {result}",
    (10, output_img.shape[0] - 10),  #pos text (x,y)
    cv2.FONT_HERSHEY_SIMPLEX, #font text
    1.5,                #mida text
    (0, 255, 0),       #color text(verd)
    2,                  #gruix text
    cv2.LINE_AA
)
cv2.imwrite("output/plate_detector_out.jpg", output_img)
plt.imshow(cv2.cvtColor(output_img, cv2.COLOR_BGR2RGB))
plt.title("Matrícula detectada en la imatge")
plt.axis('off')
plt.show()
