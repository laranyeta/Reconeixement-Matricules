############### RECONEIXEMENT DE MATRÍCULES DE VEHICLES #####################
# Autors: Pau Bofill, Lara Castillejo, Júlia Lipin Gener
# Curs: 2024-2025


import cv2
from matplotlib import pyplot as plt
from car_utils import *
import numpy as np

def extract_gt_xml_data(gt):
    return np.array([int(gt.find('object').find('bndbox').find('xmin').text),
            int(gt.find('object').find('bndbox').find('ymin').text),
            int(gt.find('object').find('bndbox').find('xmax').text),
            int(gt.find('object').find('bndbox').find('ymax').text)])
    
def getBoundingBoxError(bound, gt):
    diff = bound - gt
    diff = diff * diff 
    sumSquares= diff.sum()
    return np.sqrt(  sumSquares) 

def detect_plate(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Operación blackhat para resaltar texto oscuro sobre fondo claro
    rectKern = cv2.getStructuringElement(cv2.MORPH_RECT, (13,5))
    blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, rectKern)

    # Umbral para extraer regiones claras (potenciales placas)
    _, light = cv2.threshold(blackhat, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

    # Mejorar bordes con gradiente X
    gradX = cv2.Sobel(blackhat, cv2.CV_32F, 1, 0, ksize=-1)
    gradX = cv2.convertScaleAbs(gradX)
    gradX = cv2.GaussianBlur(gradX, (5,5), 0)
    gradX = cv2.morphologyEx(gradX, cv2.MORPH_CLOSE, rectKern)
    _, thresh = cv2.threshold(gradX, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

    # Limpiar la máscara resultante
    thresh = cv2.erode(thresh, None, iterations=2)
    thresh = cv2.dilate(thresh, None, iterations=2)
    # Encontrar contornos y filtrar por forma de matrícula
    contornos, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    plate_cnt = None
    for c in sorted(contornos, key=cv2.contourArea, reverse=True):
        x,y,w,h = cv2.boundingRect(c)
        ar = w/float(h)
        if 2.5 < ar < 5.0 and w*h > 1000:  # ajustar condiciones según escenario
            plate_cnt = c
            break
    if plate_cnt is None:
        return None
    return cv2.boundingRect(plate_cnt)

if __name__ == "__main__":
    images = read_images('database/plates/images/')
    gt = read_xml_files('database/plates/annotations/')

    errors = np.zeros(images.__len__())-1
    detected_bounds = np.zeros((images.__len__(), 4))
    for i, img in enumerate(images):
        bound = detect_plate(img[0])
        imgGt = extract_gt_xml_data(gt[i])

        imgGt = np.array(imgGt)
        if bound is not None:
            bound = np.array(bound)
            detected_bounds[i] = bound
            errors[i] = getBoundingBoxError(bound, imgGt)

    detected_errors = errors[errors != -1]
    failed = errors[errors == -1]
    print (f"Average error: {errors.mean():.2f}")
    print (f"Failed images: {failed.__len__()}")    

    max_error = errors.argmax()

    #Set the error to a high value to avoid it being selected as the best one
    forMin_error = errors
    forMin_error[forMin_error == -1] = 10000

    min_error = forMin_error.argmin()

    detected_bound_min = display_bounding_box(images[min_error][0], detected_bounds[min_error], color=(255, 255, 0))
    detected_bound_min = display_bounding_box(detected_bound_min, extract_gt_xml_data(gt[min_error]), color=(255, 255, 0))

    detected_bound_max = display_bounding_box(images[max_error][0], detected_bounds[max_error])
    detected_bound_max = display_bounding_box(detected_bound_max, extract_gt_xml_data(gt[max_error]))


    fig, ax = plt.subplots(1, 2)
    ax[0].imshow(cv2.cvtColor(detected_bound_max, cv2.COLOR_BGR2RGB))
    ax[0].set_title(f"Max error: {errors[max_error]:.2f}")
    ax[0].axis('off')
    ax[1].imshow(cv2.cvtColor(detected_bound_min, cv2.COLOR_BGR2RGB))
    ax[1].set_title(f"Min error: {errors[min_error]:.2f}")
    ax[1].axis('off')
    plt.show()
