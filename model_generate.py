import cv2


cap = cv2.VideoCapture(0)
cv2.namedWindow('Windows',cv2.WINDOW_AUTOSIZE)

faceRecognizerCascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

faceRecognizer = cv2.face.LBPHFaceRecognizer_create()
faceRecognizer.read('trainedData.yml')

scale = 1.01
neighbors = 125

while (True):

    ret, imagem = cap.read()
    faces = faceRecognizerCascade.detectMultiScale(imagem,scale,neighbors)

    for(x,y,w,h) in faces:
        faceImage = imagem[y:y+h,x:x+w]
        grayFaceImage = cv2.cvtColor(faceImage,cv2.COLOR_BGR2GRAY)
        id,confianca = faceRecognizer.predict(grayFaceImage)
        print("ID:",id)
        print("confianca:",confianca)
        cv2.rectangle(imagem,(x,y),(x+w,y+h),(255,255,0),4)


    cv2.imshow('Windows',imagem)

    if cv2.waitKey(1) == 27:
        break


cap.release()
cv2.destroyAllWindows()
