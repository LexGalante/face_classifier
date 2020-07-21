import cv2
import numpy

faceRecognizer = cv2.face.LBPHFaceRecognizer_create() 
faceList = []
faceIds = []

for x in range(0, 30):
    image = cv2.imread("./dataset-train/{}.jpg".format(str(x)), cv2.IMREAD_GRAYSCALE)
    faceList.append(image)
    faceIds.append(1)

faceRecognizer.train(faceList, numpy.array(faceIds))
faceRecognizer.write('trained_model.yml')
