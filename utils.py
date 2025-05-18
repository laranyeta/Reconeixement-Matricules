######################################
### FUNCIONS UTILS PER AL PROJECTE ###
######################################
''' Aquest script inclou funcions auxiliars
    utilitzades de forma general al projecte
'''
import cv2
import os
import xml.etree.ElementTree as ET

def read_images(path):
    images = []
    for filename in os.listdir(path):
        img_path = os.path.join(path, filename)
        img = cv2.imread(img_path)
        images.append((img, filename))
    return images

def read_xml_files(path): #lectura anotacions de les matricules
    data = []
    for filename in os.listdir(path):
        xml_path = os.path.join(path, filename)
        file = ET.parse(xml_path)
        data.append(file)
    return data

def display_bounding_box(image, box, color=(0, 255, 0)):
    print(box)
    x1, y1, x2, y2 = map(int, box)
    BoundedImage = image.copy()
    cv2.rectangle(BoundedImage, (x1, y1), (x2, y2), color=color, thickness=2)
    return BoundedImage

def crop_car(result, image):
    box = result[0].boxes[0]
    x1, y1, x2, y2 = box.xyxy[0].tolist()
    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
    cropped_img = image[y1:y2, x1:x2]
    return cropped_img

def crop_plate(bounding_box, image):
    if bounding_box is not None:
        x1, y1, x2, y2 = map(int, bounding_box)
        cropped_plate = image[y1:y2, x1:x2]
        return cropped_plate

