import cv2
import numpy as np
import function

webcam = False
path = '1.jpg'
cap = cv2.VideoCapture(1)
# cap.set(10, 160)
# cap.set(3, 1920)
# cap.set(4, 1080)
# cap.set(cv2.CAP_PROP_FRAME_WIDTH, 720)
# cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
# fourcc = cv2.VideoWriter_fourcc(*'XVID')
# cv2.VideoWriter('output.avi', fourcc, 20.0, (720, 480))
scale = 3
wp =297*scale
hp = 210*scale

while True:
    success, img = cap.read()
    if success:
        img = cv2.resize(img, (0, 0), fx=1, fy=1)
        img, conts = function.getContours(img, minArea=50000, filter=4)
        cv2.imshow('Original', img)
    else:
        img = cv2.imread(path)
        img = cv2.resize(img, (0, 0), fx=0.2, fy=0.2)
        img, conts = function.getContours(img,minArea=50000,filter=4)
        cv2.imshow('Original', img)

    if len(conts) != 0:
        biggest = conts[0][2]
        print(biggest)
        imgWarp = img = function.warpImg(img, biggest, wp, hp)
        imgContours2, conts2 = function.getContours(imgWarp, cThr=[15,15],showCanny=True, minArea=2000, filter=4)
        if len(conts) != 0:
            for obj in conts2:
                cv2.polylines(imgContours2,[obj[2]],True,(0,255,0),2)
                nPoints = function.reoder(obj[2])
                nW = round((function.findDis(nPoints[0][0]//scale,nPoints[1][0]//scale)/10),1)
                nH = round((function.findDis(nPoints[0][0]//scale,nPoints[2][0]//scale)/10), 1)

                cv2.arrowedLine(imgContours2, (nPoints[0][0][0], nPoints[0][0][1]),
                                (nPoints[1][0][0],nPoints[1][0][1]),(255, 0, 255), 3, 8, 0, 0.05)
                cv2.arrowedLine(imgContours2, (nPoints[0][0][0], nPoints[0][0][1]),
                                (nPoints[2][0][0],nPoints[2][0][1]),(255, 0, 255), 3, 8, 0, 0.05)

                x, y, w, h = obj[3]
                cv2.putText(imgContours2, '{}cm'.format(nW), (x + 30, y - 10), cv2.FONT_HERSHEY_COMPLEX_SMALL,2,
                            (255, 0, 255), 2)
                cv2.putText(imgContours2, '{}cm'.format(nH), (x - 70, y + h // 2), cv2.FONT_HERSHEY_COMPLEX_SMALL,2,
                            (255, 0, 255), 2)

        cv2.imshow('A4', imgContours2)

    if cv2.waitKey(1) == ord('q'):
        break