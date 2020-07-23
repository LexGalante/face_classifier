import cv2
import numpy


def generate():
    model = cv2.face.LBPHFaceRecognizer_create()
    face_list = []
    face_ids = []

    for x in range(1, 31):
        image = cv2.imread(f"./dataset-train/{x}.jpg", cv2.IMREAD_GRAYSCALE)
        face_recognizer = cv2.CascadeClassifier('./models/haarcascade_frontalface_default.xml')
        face = face_recognizer.detectMultiScale(image, 1.01, 125)
        if face is not None and len(face) > 0:
            for (x, y, w, h) in face:
                face_image = image[y:y + h, x:x + w]
                face_list.append(face_image)
            face_ids.append(1)

    model.train(face_list, numpy.array(face_ids))
    model.write('./models/trained_model.yml')
