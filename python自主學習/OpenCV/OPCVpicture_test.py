import cv2 as cv
import numpy as np

kernel = np.ones((3,3), np.uint8)
kernel1 = np.ones((3,3), np.uint8)
img = cv. imread('sword.jpg')
img = cv.resize(img, (0,0), fx=1 , fy=1)

gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
gauss = cv.GaussianBlur(img, (3,3), 0)
canny = cv.Canny(img, 150, 200)
dilate = cv.dilate(canny, kernel, iterations= 1)
erode = cv.erode(canny, kernel1, iterations= 1)


cv.imshow('img', img)
cv.imshow('gray', gray)
cv.imshow('gauss', gauss)
cv.imshow('canny', canny)
cv.imshow('dilate', dilate)
cv.imshow('erode', erode)
cv.waitKey(0)