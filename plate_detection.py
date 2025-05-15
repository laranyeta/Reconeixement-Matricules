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
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    equalized = cv2.equalizeHist(blurred)           
    return equalized

def preprocess_1(image):
   # 1. Read and convert
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Enhance contrast using CLAHE
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)

    # Gamma correction
    gamma = 1.5
    look_up = np.array([((i / 255.0) ** (1.0 / gamma)) * 255 for i in range(256)], dtype="uint8")
    adjusted = cv2.LUT(enhanced, look_up)

    # Denoising
    blurred = cv2.bilateralFilter(adjusted, d=9, sigmaColor=75, sigmaSpace=75)

    return blurred

def score_contour(cnt):
    x, y, w, h = cv2.boundingRect(cnt)
    aspect_ratio = w / float(h)
    area = w * h
    contour_area = cv2.contourArea(cnt)
    density = contour_area / area if area > 0 else 0

    # Define score weights
    if 2 < aspect_ratio < 6 and area > 1000:
        # Favor aspect ratio ~4 (typical for plates), high density
        score = (
            -abs(aspect_ratio - 4) * 2     # penalize deviation from ideal aspect ratio
            + density * 10                 # reward tight contours
            + min(area / 5000, 1) * 2      # reward reasonable area size
        )
        return score, (x, y, x + w, y + h)
    return -1, None


def locate_plate(image, showProcess=False):

    # 3. Perform Canny edge detection
    edges = cv2.Canny(image, 50, 150)

    # 4. Detect lines using Probabilistic Hough Transform
    lines = cv2.HoughLinesP(edges,
                            rho=1,
                            theta=np.pi / 180,
                            threshold=80,
                            minLineLength=70,
                            maxLineGap=15)
    if lines is None:
        return None

    # 5. Create a blank image to draw lines
    line_img = np.zeros_like(edges)

    # 6. Draw the detected lines on the blank image
    for x1, y1, x2, y2 in lines[:, 0]:
        cv2.line(line_img, (x1, y1), (x2, y2), 255, 2)

    # 7. Find contours from the line image
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # 8. Loop through contours to find potential license plate regions
    scores = np.zeros(len(contours))
    bounds = np.zeros((len(contours), 4))
    for i, cnt in enumerate(contours):
        score, bound = score_contour(cnt)
        scores[i] = score
        if bound is not None:
            bounds[i] = bound

    # 9. Find the contour with the highest score
    best_index = np.argmax(scores) 
    best_score = scores[best_index]
    best_bound = bounds[best_index]

    if showProcess:
        fig, ax = plt.subplots(1, 4 )

        after_canny = cv2.Canny(image, 50, 150)
        after_canny = cv2.cvtColor(after_canny, cv2.COLOR_GRAY2BGR)
        # 1. Image after Canny edge detection with hough lines
        for x1, y1, x2, y2 in lines[:, 0]:
            cv2.line(after_canny, (x1, y1), (x2, y2), (0, 255, 0), 2)

        #image with all contours 
        countours_img = image.copy()
        for cnt in contours:
            cv2.drawContours(countours_img, [cnt], -1, (0, 255, 0), 2)
        

        # 2. Image with the best bounding box
        best_img = image.copy()
        if best_bound is not None:
            x1, y1, x2, y2 = best_bound.astype(int)
            cv2.rectangle(best_img, (x1, y1), (x2, y2), (0, 255, 0), 2)

        ax[0].imshow(after_canny )
        ax[0].set_title('Canny + Hough Lines')
        ax[0].axis('off')
        ax[1].imshow(countours_img, 'gray')
        ax[1].set_title('Contours')
        ax[1].axis('off')
        ax[2].imshow(best_img, 'gray')
        ax[2].set_title('Best Bounding Box')
        ax[2].axis('off')
        ax[3].imshow(image, 'gray')
        ax[3].set_title('Original Image')
        ax[3].axis('off')

        manager = plt.get_current_fig_manager()
        try:
            manager.window.showMaximized()  # Qt5Agg backend
        except AttributeError:
            try:
                manager.window.state('zoomed')  # TkAgg backend on Windows
            except Exception as e:
                print("Maximize not supported:", e)

        plt.show()
    return best_bound if best_score > 0 else None



def detect_plate(image, debug=False):
    preprocessed = preprocess_1(image)
    plate_bounds = locate_plate(preprocessed, debug)
    return plate_bounds


def test_algorithm(save=False, debug=False):
    images = read_images('database/plates/images/')
    gt = read_xml_files('database/plates/annotations/')

    errors = np.zeros(images.__len__())-1
    detected_bounds = np.zeros((images.__len__(), 4))
    for i, img in enumerate(images):
        bound = detect_plate(img[0], debug)
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
    test_algorithm(True,False )