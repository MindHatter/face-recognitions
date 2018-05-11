# -*- coding:utf-8 -*-
import dlib
import scipy.misc
import numpy as np
import os
import sys
import time

#face detector allows us to detect faces in images
face_detector = dlib.get_frontal_face_detector()
#pose predictor allows us to detect landmark points in faces and understand the pose/angle of the face
shape_predictor = dlib.shape_predictor('./depends/shape_predictor_68_face_landmarks.dat')
# face recognition model is what gives us numbers that identify the face of a particular person)
face_recognition_model = dlib.face_recognition_model_v1('./depends/dlib_face_recognition_resnet_model_v1.dat')
# lower to avoid false matches, higher to avoid false negatives (i.e. faces of the same person doesn't match)
TOLERANCE = 0.6

#this function will take an image and return its face encodings using the neural network
def get_face_encodings(path_to_image):
    image = scipy.misc.imread(path_to_image)
    detected_faces = face_detector(image, 1)
    shapes_faces = [shape_predictor(image, face) for face in detected_faces]
    return [np.array(face_recognition_model.compute_face_descriptor(image, face_pose, 1)) for face_pose in shapes_faces]

#this function takes a list of known faces
def compare_face_encodings(known_faces, face):
    global norms
    norms = np.linalg.norm(known_faces - face, axis=1)
    return (norms <= TOLERANCE)

#this function returns the name of the person whose image matches with the given face (or 'Not Found')
def find_match(known_faces, names, face):
    matches = compare_face_encodings(known_faces, face)
    count = 0
    for match in matches:
        if match:
            return count
        count += 1
    return 'Not Found'

if __name__ == "__main__":
    #face detector allows us to detect faces in images
    face_detector = dlib.get_frontal_face_detector()
    #pose predictor allows us to detect landmark points in faces and understand the pose/angle of the face
    shape_predictor = dlib.shape_predictor('./depends/shape_predictor_68_face_landmarks.dat')
    # face recognition model is what gives us numbers that identify the face of a particular person)
    face_recognition_model = dlib.face_recognition_model_v1('./depends/dlib_face_recognition_resnet_model_v1.dat')
    # lower to avoid false matches, higher to avoid false negatives (i.e. faces of the same person doesn't match)
    TOLERANCE = 0.6

    data_folder_path = 'train_data'
    dirs = os.listdir(data_folder_path)

    face_encodings = [] 
    image_filenames = []

    print("Begin detecting and training. Waiting...")
    train_time = time.time()
    for dir_name in dirs:
        for img_name in os.listdir(os.path.join('train_data', dir_name)):
            path_to_image = os.path.join('train_data', dir_name, img_name)
            face_encodings_in_image = get_face_encodings(path_to_image)
            if len(face_encodings_in_image) != 1:
                print("Not faces in", path_to_image)
                continue
            image_filenames.append(path_to_image)
            face_encodings.append(face_encodings_in_image[0])

    train_time = round(time.time() - train_time, 3)
    print("Detecting and training time: {}m {}s".format(train_time//60, train_time%60))

    norms = []
    names = [x[:-4] for x in image_filenames]
    path_to_test_image = os.path.join('test_data', sys.argv[1])
    
    print("Begin recognizing...")
    reco_time = time.time()
    face_encodings_in_image = get_face_encodings(path_to_test_image)
    match = find_match(face_encodings, names, face_encodings_in_image[0])
    reco_time = round(time.time() - reco_time, 3)
    print("They are {} with {} confidence".format(names[match], 1-norms[match]))
    print("Recognizing time: {}m {}s".format(reco_time//60, reco_time%60))
    
    total_time = round(train_time + reco_time, 3)
    print("Total time: {}m {}s".format(total_time//60, total_time%60))
