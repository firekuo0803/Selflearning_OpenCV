import cv2
import time
import HandFunctions as hf

pTime = 0
cTime = 0
cam = cv2.VideoCapture(1)
detector = hf.handDetector()
while True:
    ret, img = cam.read()
    img = detector.findHands(img)
    lmList = detector.findPosition(img)
    if len(lmList) != 0:
        print(lmList[4])

    img = cv2.resize(img, None, fx=1, fy=1)

    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime
    cv2.putText(img, f"FPS:{int(fps)}", (30, 50), cv2.FONT_HERSHEY_COMPLEX_SMALL, 2, (255, 0, 0), 2)

    cv2.imshow('img', img)
    if cv2.waitKey(1) == ord('q'):
        break