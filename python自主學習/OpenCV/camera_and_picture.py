import cv2
import numpy as np

# img = cv2.imread('sword.jpg')
#
# img = cv2.resize(img, (0,0), fx= 1, fy = 1)
# cv2.imshow('img', img)
# cv2.waitKey(0)

cap = cv2.VideoCapture(1)

while True:
    ret, frame = cap.read()

    if ret:
        frame = cv2.resize(frame, (0,0), fx=1, fy=1)
        cv2.imshow('video', frame)
    else:
        break
    cv2.waitKey(1)



