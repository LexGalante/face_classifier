import cv2
import os


def classify():
    list_images = os.listdir("./dataset-twitter")
    face_recognizer_cascade = cv2.CascadeClassifier('./models/haarcascade_frontalface_default.xml')
    face_recognizer = cv2.face.LBPHFaceRecognizer_create()
    face_recognizer.read('./models/trained_model.yml')

    scale = 1.01
    neighbors = 125
    match = 0

    for i in range(0, len(list_images)):
        image = cv2.imread(f"./dataset-twitter/{i}.jpg", cv2.COLOR_BGR2GRAY)
        face = face_recognizer_cascade.detectMultiScale(image, scale, neighbors)
        if face is not None and len(face) > 0:
            for(x, y, w, h) in face:
                face_image = image[y:y + h, x:x + w]
                roi = cv2.cvtColor(face_image, cv2.COLOR_BGR2GRAY)
                id, accuracy = face_recognizer.predict(roi)
                if accuracy < 50:
                    match = match + 1

    x = (match / len(list_images))
    return x

