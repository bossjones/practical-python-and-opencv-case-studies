# pylint:disable=no-member
# python -m practical_python_and_opencv_case_studies.faces.faces_train

import os
import pathlib
import time

import cv2 as cv
import numpy as np
from typing import Union, Any

ROOT_DIR = os.path.dirname(__file__)

photos_dir = pathlib.Path(
    f"/Users/malcolm/dev/bossjones/practical-python-and-opencv-case-studies/images/resources/faces/train"
)


people = ["Ben Afflek", "Elton John", "Jerry Seinfield", "Madonna", "Mindy Kaling"]
DIR = f"{photos_dir}"

haar_cascade = cv.CascadeClassifier(f"{ROOT_DIR}/haar_face.xml")

# image array of faces
features = []
# labels of who's face it is
labels = []


def create_train():
    for person in people:
        path = os.path.join(DIR, person)
        label = people.index(person)

        for img in os.listdir(path):
            img_path = os.path.join(path, img)

            img_array: Union[np.ndarray, Any]
            img_array = cv.imread(img_path)
            if img_array is None:
                continue

            gray: Union[np.ndarray, Any]
            gray = cv.cvtColor(img_array, cv.COLOR_BGR2GRAY)

            # find faces or eyes
            # Definition: cv2.CascadeClassifier.detectMultiScale(image[, scaleFactor[, minNeighbors[, flags[, minSize[, maxSize]]]]])
            # image : Matrix of the type CV_8U containing an image where objects are detected.
            # scaleFactor : Parameter specifying how much the image size is reduced at each image scale.
            # minNeighbors: Parameter specifying how many neighbors each candidate rectangle should have to retain it. This parameter will affect the quality of the detected faces: higher value results in less detections but with higher quality. We're using 5 in the code.
            # flags: Parameter with the same meaning for an old cascade as in the function cvHaarDetectObjects. It is not used for a new cascade.
            # minSize : Minimum possible object size. Objects smaller than that are ignored.
            # maxSize : Maximum possible object size. Objects larger than that are ignored.
            faces_rect = haar_cascade.detectMultiScale(
                gray, scaleFactor=1.1, minNeighbors=4
            )

            for (x, y, w, h) in faces_rect:
                # crop out the face in the image
                faces_roi = gray[y : y + h, x : x + w]
                features.append(faces_roi)
                labels.append(label)


create_train()
print("Training done ---------------")

features = np.array(features, dtype="object")
labels = np.array(labels)

face_recognizer: cv.face_LBPHFaceRecognizer
face_recognizer = cv.face.LBPHFaceRecognizer_create()
print(f"{type(face_recognizer)}")

print("[INFO] training face recognizer...")
start = time.time()
# Train the Recognizer on the features list and the labels list
face_recognizer.train(features, labels)
end = time.time()
print("[INFO] training took {:.4f} seconds".format(end - start))

# save the model so we can use it in other files as well
face_recognizer.save(f"{ROOT_DIR}/face_trained.yml")
np.save(f"{ROOT_DIR}/features.npy", features)
np.save(f"{ROOT_DIR}/labels.npy", labels)
