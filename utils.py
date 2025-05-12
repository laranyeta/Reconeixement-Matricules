import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import xml.etree.ElementTree as ET

def read_images(path):
    images = []
    for filename in os.listdir(path):
        img_path = os.path.join(path, filename)
        img = cv2.imread(img_path)
        images.append((img, filename))
    return images

def read_xml_files(path):
    data = []
    for filename in os.listdir(path):
        xml_path = os.path.join(path, filename)
        file = ET.parse(xml_path)
        data.append(file)
    return data

def read_video(path):
    video = cv2.VideoCapture(path)
    frames = []

    while True: #mentre hi hagi frames -> extraiem frames
        flag, frame_bgr = video.read() #flag indica si hi ha frames (1) o no (0)
        frame_bw = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        if not flag:
            break
        frames.append(frame_bw)
    video.release()
    return frames

def display_bounding_box(image, box, color=(0, 255, 0)):
    print(box)
    x1, y1, x2, y2 = box.astype(int) 
    BoundedImage = image.copy()
    cv2.rectangle(BoundedImage, (x1, y1), (x2, y2), color=color, thickness=2)
    return BoundedImage

def plate_to_text(model, results):
    boxes = results[0].boxes
    class_ids = boxes.cls.cpu().numpy()
    coords = boxes.xyxy.cpu().numpy()
    names = model.names #retorna el nom de cada classe

    detections = [] #llista de caracters detectats
    for cls_id, coord in zip(class_ids, coords):
        x1 = coord[0]
        char = names[int(cls_id)]
        detections.append((x1, char))
    detections.sort(key=lambda x: x[0]) #ordenar per coordenada x (horitzontal)
    plate_text = ''.join([char for _, char in detections]) #reconstruir matricula
    return plate_text
    
