# coding: utf-8

import cv2
import os
import time
import sys
import numpy as np


# function detects face using OpenCV
def detect_face(img):
    #convert the test image to gray image as opencv face detector expects gray images
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    #choose Haar or LBP cascade
    face_cascade = cv2.CascadeClassifier('./depends/haarcascade.xml')

    #detect multiscale (some images may be closer to camera than others) images to a list of faces
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5)

    #if no faces are detected then return original img
    if (len(faces) == 0):
        return None, None

    #extract the face area and return only the face part of the image
    (x, y, w, h) = faces[0]
    return gray[y:y+w, x:x+h], faces[0]


def prepare_training_data(data_folder_path):
    dirs = os.listdir(data_folder_path)
    faces = []
    labels = []

    # read all images and find faces
    for dir_name in dirs:
        if not dir_name.startswith("s"):
            continue;

        label = int(dir_name.replace("s", ""))
        subject_dir_path = data_folder_path + "/" + dir_name
        subject_images_names = os.listdir(subject_dir_path)

        for image_name in subject_images_names:
            #ignore system files
            if image_name.startswith("."):
                continue;

            image_path = subject_dir_path + "/" + image_name
            image = cv2.imread(image_path)
            face, rect = detect_face(image)

            if face is not None:
                faces.append(face)
                labels.append(label)

    return faces, labels

def predict(test_img):
    img = test_img.copy()
    face, rect = detect_face(img)
    (x, y, w, h) = rect
    cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
    return face_recognizer.predict(face)


if __name__ == "__main__":
    # there is no label 0 in our training data so subject name for index/label 0 is empty
    subjects = [""]
    subjects += [str(i) for i in range(1, len(os.listdir('./train_data'))+1)]
    face_recognizer = cv2.face.LBPHFaceRecognizer_create()

    print("Begin detecting faces. Waiting...")
    detect_time = time.time()
    faces, labels = prepare_training_data("train_data")
    detect_time = round(time.time()- detect_time,3)
    print("Detecting time: {}m {}s".format(detect_time//60, detect_time%60))

    print("Begin training")
    train_time = time.time()
    face_recognizer.train(faces, np.array(labels))
    train_time = round(time.time() - train_time, 3)
    print("Training time: {}m {}s".format(train_time//60, train_time%60))

    test_img = cv2.imread("test_data/"+sys.argv[1])

    print("Begin recognizing")
    reco_time = time.time()
    label, confidence = predict(test_img)
    reco_time = round(time.time() - reco_time, 3)
    print("They are {} with {} confidence".format(label, confidence))
    print("Recognizing time: {}m {}s".format(reco_time//60, reco_time%60))

    total_time = train_time + detect_time +reco_time
    print("Total time: {}m {}s".format(total_time//60, total_time%60))
