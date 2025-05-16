###########################
### OCR (DEEP LEARNING) ###
###########################

###########################
### OCR (DEEP LEARNING) ###
###########################

import os
import cv2
import numpy as np
import pickle
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# FASE TRAINING (només realitzar un cop)
'''
dataset_path = "database/plate_ocr/chars74k/"
print(f"Ruta dataset: {dataset_path}") 
print("Contenido en dataset_path:", os.listdir(dataset_path))

images = []
labels = []

for class_name in os.listdir(dataset_path):
    class_dir = os.path.join(dataset_path, class_name)
    if not os.path.isdir(class_dir):
        continue
    print(f"Llegint classe: {class_name}")
    for img_name in os.listdir(class_dir):
        img_path = os.path.join(class_dir, img_name)
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue
        img = cv2.resize(img, (32, 32))
        images.append(img)
        labels.append(class_name)

print(f"Total imatges carregades: {len(images)}")
print(f"Total labels carregades: {len(labels)}")

images = np.array(images) / 255.0
images = images.reshape(-1, 32, 32, 1)

label_encoder = LabelEncoder()
labels_encoded = label_encoder.fit_transform(labels)
labels_categorical = to_categorical(labels_encoded)

with open("encoders/label_encoder.pkl", "wb") as f:
    pickle.dump(label_encoder, f)

X_train, X_test, y_train, y_test = train_test_split(images, labels_categorical, test_size=0.2, random_state=42)

datagen = ImageDataGenerator(
    rotation_range=5,
    width_shift_range=0.05,
    height_shift_range=0.05,
    zoom_range=0.1,
    shear_range=2,
    fill_mode='nearest'
)

cnn = Sequential([
    Conv2D(64, (3,3), activation='relu', input_shape=(32,32,1)),
    MaxPooling2D((2,2)),
    Conv2D(128, (3,3), activation='relu'),
    MaxPooling2D((2,2)),
    Flatten(),
    Dense(256, activation='relu'),
    Dropout(0.5),
    Dense(len(label_encoder.classes_), activation='softmax')
])

cnn.compile(optimizer=Adam(), loss='categorical_crossentropy', metrics=['accuracy'])

print("Training...")
cnn.fit(datagen.flow(X_train, y_train, batch_size=16),
        epochs=20,
        validation_data=(X_test, y_test))

cnn.save("models/cnn_plate.h5")
'''
model = load_model("models/cnn_plate.h5")
with open("label_encoder.pkl", "rb") as f:
    label_encoder = pickle.load(f)

def preprocess(img):
    img = cv2.resize(img, (32, 32))
    img = img / 255.0
    return img.reshape(1, 32, 32, 1)

def predict_char(img):
    x = preprocess(img)
    pred = model.predict(x)
    label_idx = np.argmax(pred)
    return label_encoder.inverse_transform([label_idx])[0]

def segment_characters(plate_img):
    gray = cv2.cvtColor(plate_img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5,5), 0)
    _, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    char_regions = []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        if 15 < h < 100 and 5 < w < 80 and h > w:
            char_regions.append((x, y, w, h))

    char_regions = sorted(char_regions, key=lambda b: b[0])
    
    char_images = []
    for x, y, w, h in char_regions:
        char = thresh[y:y+h, x:x+w]
        char = cv2.resize(char, (32,32))
        char = 255 - char
        char_images.append(char)
    
    return char_images

def predict_plate_text(plate_img):
    char_images = segment_characters(plate_img)
    for i, char_img in enumerate(char_images):
        cv2.imshow(f"Caracter {i}", char_img)
        cv2.waitKey(0)
    cv2.destroyAllWindows()

    text = ""
    for char_img in char_images:
        processed = preprocess(char_img)
        prediction = model.predict(processed)
        predicted_label = label_encoder.inverse_transform([np.argmax(prediction)])[0]
        text += predicted_label
    return text
