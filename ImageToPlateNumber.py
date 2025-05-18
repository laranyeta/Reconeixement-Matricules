from car_detection import *
from plate_detection import *
from plate_ocr_recog import *
from utils import *

class ImageToPlateNumber: 
    def __init__(self, image_path, mode = "silent"):
        self.image_path = image_path
        self.mode = mode
        self.images = read_images(image_path)
        self.CarDetector = YOLO("car_yolo/runs/train_fast/yolov8n_cotxes2/weights/best.pt")

    def __init__(self,image, mode = "silent"):
        self.images = [image]
        self.mode = mode
        self.image_path = None
        self.CarDetector = YOLO("car_yolo/runs/train_fast/yolov8n_cotxes2/weights/best.pt")

    def __init__(self, image_path, data_path,  mode = "silent"):
        self.image_path = image_path
        self.mode = mode
        self.images = get_images(image_path)
        self.database_path = data_path
        self.CarDetector = YOLO("car_yolo/runs/train_fast/yolov8n_cotxes2/weights/best.pt")

    def printModelPerfomance():
        return

    def evaluateModel(self, database_path, model_path):
        return

    def detectCar(self, index = -1):
        if index == -1:
            index = 0
        image = self.images[index]
        image_resized = cv2.resize(image, image.shape / 2)
        result = self.CarDetector.predict(image_resized, save=True, imgsz=image.shape)
        cropped_car = crop_car(result, image)
        return cropped_car,result

    def detectPlate(self, img):
        plate = detect_plate(img)
        croped_plate = crop_plate(plate, img)
        return croped_plate

    def predictPlateNumber(self, plate):
        result, char_imgs, char_preds = predict_plate_text(plate)
        return result, char_imgs, char_preds

    def saveStages(self, image, car, plate):
        cv2.imwrite("output/img/cropped_car.jpg", car)
        cv2.imwrite("output/img/cropped_plate.jpg", plate)
        return

    def processImage(self, index = -1):
        car,cropping_result = self.detectCar(index)
        plate = self.detectPlate(index, car)
        plate_number = self.predictPlateNumber(index)
        return plate_number