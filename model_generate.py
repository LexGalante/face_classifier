import cv2
import numpy
from face_classifier.classifier import classify


def generate():
    faceRecognizer = cv2.face.LBPHFaceRecognizer_create()
    faceList = []
    faceIds = []

    for x in range(1, 30):
        image = cv2.imread(f"./dataset-train/{x}.jpg", cv2.IMREAD_GRAYSCALE)
        faceList.append(image)
        faceIds.append(1)

    faceRecognizer.train(faceList, numpy.array(faceIds))
    faceRecognizer.write('./models/trained_model.yml')

    percent = classify()
    return percent
