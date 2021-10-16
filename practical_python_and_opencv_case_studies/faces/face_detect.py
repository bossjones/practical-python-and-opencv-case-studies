# pylint:disable=no-member
# https://github.com/jasmcaus/opencv-course

import os
import pathlib

import cv2 as cv

ROOT_DIR = os.path.dirname(__file__)

photos_dir = pathlib.Path(
    f"/Users/malcolm/dev/bossjones/practical-python-and-opencv-case-studies/images/resouces/photos"
)

img = cv.imread(f"{photos_dir}/group 1.jpg")
cv.imshow("Group of 5 people", img)

gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
cv.imshow("Gray People", gray)

haar_cascade = cv.CascadeClassifier(f"{ROOT_DIR}/haar_face.xml")

faces_rect = haar_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=1)

print(f"Number of faces found = {len(faces_rect)}")

for (x, y, w, h) in faces_rect:
    cv.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), thickness=2)

cv.imshow("Detected Faces", img)


cv.waitKey(0)
