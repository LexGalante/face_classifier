import cv2
import numpy
import os

faceRecognizer = cv2.face.LBPHFaceRecognizer_create() 
faceList=[]
faceIds=[]
for x in range(0,99):
    image = cv2.imread("./Diego/imagem%d.jpg" %x,cv2.IMREAD_GRAYSCALE)
    faceList.append(image)
    faceIds.append(1)

faceRecognizer.train(faceList,numpy.array(faceIds))
faceRecognizer.write('trainedData.yml')
