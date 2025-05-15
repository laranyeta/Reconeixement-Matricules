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
import cv2
import numpy as np

# 1. PREPROCESADO
def preprocess(image):
    # 1.1 Escala de grises
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # :contentReference[oaicite:10]{index=10}
    # 1.2 Suavizado Gaussian
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)     # :contentReference[oaicite:11]{index=11}
    # 1.3 Ecualización de histograma
    equalized = cv2.equalizeHist(blurred)           # :contentReference[oaicite:12]{index=12}
    return equalized

# 2. LOCALIZACIÓN DE MATRÍCULA
def locate_plate(pre):
   import cv2
import numpy as np

def locate_plate(image):

    # 3. Perform Canny edge detection
    edges = cv2.Canny(image, 50, 150)

    # 4. Detect lines using Probabilistic Hough Transform
    lines = cv2.HoughLinesP(edges,
                            rho=1,
                            theta=np.pi / 180,
                            threshold=50,
                            minLineLength=50,
                            maxLineGap=10)
    if lines is None:
        return None

    # 5. Create a blank image to draw lines
    line_img = np.zeros_like(edges)

    # 6. Draw the detected lines on the blank image
    for x1, y1, x2, y2 in lines[:, 0]:
        cv2.line(line_img, (x1, y1), (x2, y2), 255, 2)

    # 7. Find contours from the line image
    contours, _ = cv2.findContours(line_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # 8. Loop through contours to find potential license plate regions
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        aspect_ratio = w / float(h)
        area = w * h
        if 2 < aspect_ratio < 6 and area > 1000:
            return np.array([x, y, x + w, y + h])  # Return the bounding box coordinates

    return None


def detect_plate(image, debug=False):
    preprocessed = preprocess(image)
    plate_bounds = locate_plate(preprocessed)
    return plate_bounds


def test_algorithm(save=False):
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
            if save:
                save_test_results(img[0], bound, imgGt, errors[i])  

    detected_errors = errors[errors != -1]
    failed = errors[errors == -1]
    print (f"Average error: {errors.mean():.2f}")
    print (f"Failed images: {failed.__len__() // errors.__len__() * 100:.2f}%")    

    max_error = errors.argmax()

    #Set the error to a high value to avoid it being selected as the best one
    forMin_error = errors
    forMin_error[forMin_error == -1] = 10000

    min_error = forMin_error.argmin()

    detected_bound_min = display_bounding_box(images[min_error][0], detected_bounds[min_error], color=(255, 0, 0))
    detected_bound_min = display_bounding_box(detected_bound_min, extract_gt_xml_data(gt[min_error]), color=(0, 255, 0))

    detected_bound_max = display_bounding_box(images[max_error][0], detected_bounds[max_error], color=(255, 0, 0))
    detected_bound_max = display_bounding_box(detected_bound_max, extract_gt_xml_data(gt[max_error]), color=(0, 255, 0))


    fig, ax = plt.subplots(1, 2)
    ax[0].imshow(cv2.cvtColor(detected_bound_max, cv2.COLOR_BGR2RGB))
    ax[0].set_title(f"Max error: {errors[max_error]:.2f}")
    ax[0].axis('off')
    ax[1].imshow(cv2.cvtColor(detected_bound_min, cv2.COLOR_BGR2RGB))
    ax[1].set_title(f"Min error: {errors[min_error]:.2f}")
    ax[1].axis('off')
    plt.show()

def save_test_results(image, detected, gt, error):
    detected = display_bounding_box(image, detected, color=(255, 0, 0))
    detected = display_bounding_box(detected, gt, color=(0, 255, 0))
    cv2.imwrite(f"test_results/{error:.2f}.jpg", detected)
    print(f"Saved test result with error: {error:.2f}")

if __name__ == "__main__":
    # images = read_images('database/plates/images/')
    # gt = read_xml_files('database/plates/annotations/')

    # print(images.__len__())
    # bondingBox = detect_plate(images[412][0], debug=True)
    test_algorithm(True)