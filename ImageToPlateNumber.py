from car_detection import *
from plate_detection import *
from plate_ocr_recog import *
from utils import *

class ImageToPlateNumber: 
    def __init__(self, image_path, mode = "silent"):
        self.image_path = image_path
        self.mode = mode
        self.images = get_images(image_path)
        self.CarDetector = YOLO("car_yolo/runs/train_fast/yolov8n_cotxes2/weights/best.pt")


    def printModelPerfomance():

        return

    def evaluateModel(self, labels):
        
        labels = read_txt_files(labels)
        if len(labels) != len(self.images):
            print("Error: El nombre de imatges no coincideix amb el nombre d'etiquetes")
            return
        matricules_detectades = 0
        for i, image in enumerate(self.images):
            
            matricula  = self.predict(image)
            if matricula != None and matricula == labels[i]:
                print(f"Imatge {i} coincideix: {matricula} == {labels[i]}")
                matricules_detectades += 1
            else:
                print(f"Imatge {i} no coincideix: {matricula} != {labels[i]}")
        print(f"Precision : {matricules_detectades}/{len(labels)}")
        return

    def detectCar(self, image):
        result = self.CarDetector.predict(image,imgsz=(image.shape[0], image.shape[1]))
        if result[0].boxes is None or len(result[0].boxes) == 0:
            print("No s'ha detectat cap cotxe")
            return None,None
        cropped_car = crop_car(result, image)
        return cropped_car,result

    def detectPlate(self, img):
        plate = detect_plate(img)
        croped_plate = crop_plate(plate, img)
        return croped_plate

    def predictPlateNumber(self, plate):
        result, char_imgs, char_preds = predict_plate_text(plate)
        return result, char_imgs, char_preds


    def predict(self, image): 
        car,cropping_result = self.detectCar(image)
        if car is None:
            print("No s'ha detectat cap cotxe")
            return None
        plate = self.detectPlate(car)
        if plate is None:
            print("No s'ha detectat cap matricula")
            return None
        plate_number,_,_ = self.predictPlateNumber(plate)
        return plate_number

    def processImage(self, index = -1):
        image = self.images[index]
        car,cropping_result = self.detectCar(image)
        plate = self.detectPlate(car)
        plate_number = self.predictPlateNumber(plate)
        return plate_number

def create_test_images(dataset_path, output_path):
    images = read_images(dataset_path)
    for i, image in enumerate(images):
        filename = image[1]
        print(f"Imatge {i}: {filename}")
        plt.imshow(cv2.cvtColor(image[0], cv2.COLOR_BGR2RGB))
        plt.axis('off')
        plt.show()
        input_matricula = input("Introdueix la matricula de la imatge: ")

        newfilename = filename.split(".")[0] + ".txt"
        with open(os.path.join(output_path, newfilename), 'x1336FLG') as f:
            f.write(input_matricula)
        print(f"Matricula {input_matricula} guardada com {newfilename}")


    


def main():
    image_toplate = ImageToPlateNumber("datasets/imagetoplate/images/")
    image_toplate.evaluateModel("datasets/imagetoplate/annotations/")

if __name__ == "__main__":
    
    main()
#    create_test_images("datasets/imagetoplate/images/", "datasets/imagetoplate/annotations/")