# USAGE
# python detect_faces.py --face cascades/haarcascade_frontalface_default.xml --image images/obama.png
# python -m practical_python_and_opencv_case_studies.face_detection.detect_faces --face cascades/haarcascade_frontalface_default.xml --image images/obama.png

# import the necessary packages

import argparse

from typing import Any, Union

import bpdb
import cv2
import numpy as np

from practical_python_and_opencv_case_studies.face_detection.pyimagesearch.facedetector import (
    FaceDetector,
)

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument(
    "-f", "--face", required=True, help="path to where the face cascade resides"
)
ap.add_argument(
    "-i", "--image", required=True, help="path to where the image file resides"
)
args = vars(ap.parse_args())

# load the image and convert it to grayscale
image: Union[np.ndarray, Any]
gray: Union[np.ndarray, Any]

image = cv2.imread(args["image"])
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# find faces in the image
fd: FaceDetector

fd = FaceDetector(args["face"])

#   scaleFactor: How much the image size is reduced at
# each image scale. This value is used to create the scale
# pyramid in order to detect faces at multiple scales
# in the image (some faces may be closer to the foreground,
# and thus be larger; other faces may be smaller
# and in the background, thus the usage of varying
# scales). A value of 1.05 indicates that Jeremy is reducing
# the size of the image by 5% at each level in the
# pyramid.
#   minNeighbors: How many neighbors each window
# should have for the area in the window to be considered
# a face. The cascade classifier will detect multiple

# windows around a face. This parameter controls how
# many rectangles (neighbors) need to be detected for
# the window to be labeled a face.
#   minSize: A tuple of width and height (in pixels) indicating
# the minimum size of the window. Bounding
# boxes smaller than this size are ignored. It is a good
# idea to start with (30, 30) and fine-tune from there.
faceRects: np.ndarray
faceRects = fd.detect(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
print("I found {} face(s)".format(len(faceRects)))


bpdb.set_trace()

# loop over the faces and draw a rectangle around each
for (x, y, w, h) in faceRects:
    print(
        f"cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2) = cv2.rectangle(image, {(x, y)}, {(x + w, y + h)}, {(0, 255, 0)}, 2)"
    )
    cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

# show the detected faces
cv2.imshow("Faces", image)
cv2.waitKey(0)
