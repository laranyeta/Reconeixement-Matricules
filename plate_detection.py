#########################################
### LOCALITZACIÓ MATRICULES (CLASSIC) ###
#########################################
''' Aquest script inclou
    funcions utilitzades en aquesta fase 
'''
import cv2
from matplotlib import pyplot as plt
from utils import *
import numpy as np

def extract_gt_xml_data(gt):
    return np.array([int(gt.find('object').find('bndbox').find('xmin').text),
            int(gt.find('object').find('bndbox').find('ymin').text),
            int(gt.find('object').find('bndbox').find('xmax').text),
            int(gt.find('object').find('bndbox').find('ymax').text)])
    
def getBoundingBoxScore(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    interArea = max(0, xB - xA) * max(0, yB - yA)

    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])

    iou = interArea / float(boxAArea + boxBArea - interArea + 1e-6)
    return iou
def getBoundingBoxError(bound, gt):
    diff = bound - gt
    diff = diff * diff 
    sumSquares= diff.sum()
    return np.sqrt(  sumSquares) 

# PREPROCESSAT
def preprocess(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)) #enfatitzar contrast
    enhanced = clahe.apply(gray)

    gamma = 1.5
    look_up = np.array([((i / 255.0) ** (1.0 / gamma)) * 255 for i in range(256)], dtype="uint8")
    adjusted = cv2.LUT(enhanced, look_up) #correccio gamma

    blurred = cv2.bilateralFilter(adjusted, d=9, sigmaColor=75, sigmaSpace=75) #treure soroll a la img

    return blurred

# CALCULAR SCORE DELS CONTORNS
def score_contour(cnt):
    x, y, w, h = cv2.boundingRect(cnt)
    aspect_ratio = w / float(h)
    area = w * h
    contour_area = cv2.contourArea(cnt)
    density = contour_area / area if area > 0 else 0

    if 2 < aspect_ratio < 6 and area > 1000:
        score = (
            -abs(aspect_ratio - 4) * 3     #penalitza -> ratio no coincideix
            + density * 10                 #guanya -> contorns marcats
            + min(area / 5000, 1) * 2      #guanya -> area considerable
        )
        return score, (x, y, x + w, y + h)
    return -1, None

def preprocess_sobel(image): #treballa millor en contrasts!
    if len(image.shape) == 3 and image.shape[2] == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()

    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3) #aplicar sobel en vertical (negre sobre blanc)
    abs_sobelx = np.absolute(sobelx)
    scaled = np.uint8(255 * abs_sobelx / np.max(abs_sobelx))

    _, thr = cv2.threshold(scaled, 50, 255, cv2.THRESH_BINARY)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (17, 5)) #apliquem morfo matematica (close per unificar contorns)
    closed = cv2.morphologyEx(thr, cv2.MORPH_CLOSE, kernel)

    return closed

#LOCALITZAR MATRICULA (APLICANT SOBEL)
def locate_plate(image, showProcess=False):
    edges = preprocess_sobel(image)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    scores = np.zeros(len(contours))
    bounds = np.zeros((len(contours), 4))
    for i, cnt in enumerate(contours):
        score, bound = score_contour(cnt)
        scores[i] = score
        if bound is not None:
            bounds[i] = bound

    best_index = np.argmax(scores) 
    best_score = scores[best_index]
    best_bound = bounds[best_index]

    if showProcess:
        fig, ax = plt.subplots(1, 4)

        sobel_viz = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)

        contours_img = image.copy()
        for cnt in contours:
            cv2.drawContours(contours_img, [cnt], -1, (0, 255, 0), 2) #dibuixa contorns de la matricula

        best_img = image.copy()
        if best_bound is not None:
            x1, y1, x2, y2 = best_bound.astype(int)
            cv2.rectangle(best_img, (x1, y1), (x2, y2), (0, 255, 0), 2) #agafa el millor bounding box

        ax[0].imshow(sobel_viz)
        ax[0].set_title('Sobel + Morph')
        ax[0].axis('off')
        ax[1].imshow(contours_img, 'gray')
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
            manager.window.showMaximized()
        except AttributeError:
            try:
                manager.window.state('zoomed')
            except Exception as e:
                print("Maximize not supported:", e)

        plt.show()

    return best_bound if best_score > 0 else None #nomes retorna si hi ha un bounding box acceptable

# DETECCIÓ MATRICULA (cridant funcions anteriors)
def detect_plate(image, debug=False):
    preprocessed = preprocess(image)
    plate_bounds = locate_plate(preprocessed, debug)
    return plate_bounds

# FUNCIO DE TEST DE L'ALGORISME CREAT
def test_algorithm(save=False, debug=False):
    images = read_images('database/plates/images/')
    gt = read_xml_files('database/plates/annotations/')

    scores= np.zeros(images.__len__())-1
    errors = np.zeros(images.__len__())-1
    detected_bounds = np.zeros((images.__len__(), 4))
    for i, img in enumerate(images):
        bound = detect_plate(img[0], debug)
        imgGt = extract_gt_xml_data(gt[i])

        imgGt = np.array(imgGt)
        if bound is not None:
            bound = np.array(bound)
            detected_bounds[i] = bound
            errors[i] =  getBoundingBoxError(bound, imgGt)
            scores[i] = getBoundingBoxScore(bound, imgGt)
            if save:
                save_test_results(img[0], bound, imgGt, errors[i],i)  

    detected_errors = errors[errors != -1]
    failed = scores != 0

    errors_clean = errors[failed]

    print (f"Error mitja: {errors_clean.mean():.2f}")
    print (f"Imatges fallades: {failed.sum()}")    

    max_error = errors.argmax()

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

def save_test_results(image, detected, gt, error,index):
    detected = display_bounding_box(image, detected, color=(255, 0, 0))
    detected = display_bounding_box(detected, gt, color=(0, 255, 0))
    cv2.imwrite(f"test_results/{error:.2f}-image{index}.jpg", detected)
    print(f"S'ha guardat el test amb error: {error:.2f}")

# PROVA DE TEST
if __name__ == "__main__":
    # images = read_images('database/plates/images/')
    # gt = read_xml_files('database/plates/annotations/')

    # print(images.__len__())
    # bondingBox = detect_plate(images[412][0], debug=True)
    test_algorithm(True,False )