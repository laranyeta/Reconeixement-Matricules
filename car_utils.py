
import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

### PROVA: READ IMAGES ###
def read_images(path):
    images = []
    for filename in os.listdir(path):
        img_path = os.path.join(path, filename)
        img = cv2.imread(img_path)
        images.append((img, filename))
    return images
##########################

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

def detect_car(frames, model):
    detections = []
    for frame in frames:
        results = model(frame)[0] #results -> objecte que guarda deteccions (si hi ha box -> s'ha detectat cotxe)
        detection_boxes = [] #evitem que es sobreescriguin les coords
        for box in results.boxes: #iteracio per cada deteccio
            conf = float(box.conf) #conf -> confiança (>50% per a reconeixer com a cotxe)
            if conf > 0.5:
                x1,y1,x2,y2 = box.xyxy[0] #troba les coordenades de cada cantonada del box
                cv2.rectangle(frame, (x1,y1), (x2,y2), color = (0, 255, 0), thickness=2)
                detection_boxes.append((x1,y1,x2,y2))
        detections.append(detection_boxes)
    return detections


    
