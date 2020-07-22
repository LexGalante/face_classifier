import cv2
import os


def get_images(*args, **kwargs):
    """
    Retrieve images after download on twitter
    """
    return os.listdir("./dataset-twitter")


def get_scale_gray_image(path, *args, **kwargs):
    """ 
    Use OPENCV to read image and parse for gray scale
    """
    return cv2.imread(path, cv2.COLOR_BGR2GRAY)


def image_contains_face(face, *args, **kwargs):
    """ 
    Validate if contains a face
    """
    return face is not None and len(face) > 0


def get_roi_from_image(image, *args, **kwargs):
    """ 
    Retrieve a region of interesting
    """
    face_image = image[y:y + h, x:x + w]
    return cv2.cvtColor(face_image, cv2.COLOR_BGR2GRAY)


def get_trained_model():
    model = cv2.CascadeClassifier(
        './models/haarcascade_frontalface_default.xml')
    model = cv2.face.LBPHFaceRecognizer_create()
    model.read('./models/trained_model.yml')

    return model


def classify(acurracy=50, *args, **kwargs):
    list_images = get_images()
    model = get_trained_model()

    scale = 1.01
    neighbors = 125
    match = 0

    for i in range(0, len(list_images)):
        image = get_scale_gray_image(f"./dataset-twitter/{i}.jpg")
        face = model.detectMultiScale(
            image, scale, neighbors)
        if image_contains_face(face):
            for(x, y, w, h) in face:
                roi = get_roi_from_image(image)
                id, accuracy = face_recognizer.predict(roi)
                if accuracy < 50:
                    match = match + 1

    x = (match / len(list_images))
    return x
