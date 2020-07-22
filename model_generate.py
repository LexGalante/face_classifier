import cv2
import numpy


def generate():
    model = cv2.face.LBPHFaceRecognizer_create()
    face_list = []
    face_ids = []

    for x in range(1, 30):
        image = cv2.imread(f"./dataset-train/{x}.jpg", cv2.IMREAD_GRAYSCALE)
        face_list.append(image)
        face_ids.append(1)

    model.train(face_list, numpy.array(face_ids))
    model.write('./models/trained_model.yml')
