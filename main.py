# importing all the necessary packages
from imutils import paths
import face_recognition
import argparse
import pickle
import os
import cv2
import matplotlib.pyplot as plt

# reading the image containing a face
img1 = cv2.imread("bp3.jpg")
gray_img = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
cv2.imshow("Original_grayscale_image", gray_img)
cv2.waitKey(0)

# using Haarcascade for frontal face detection
# haar_face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_alt2.xml')
haar_face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_alt2.xml')
face = haar_face_cascade.detectMultiScale(gray_img, scaleFactor=1.1, minNeighbors=5)
for (x, y, w, h) in face:
    cv2.rectangle(img1, (x, y), (x + w, y + h), (0, 255, 0), 2)

# output the detected face in loaded image
cv2.imshow("Final_image", cv2.COLOR_BGR2RGB(img1))
cv2.waitKey(0)
