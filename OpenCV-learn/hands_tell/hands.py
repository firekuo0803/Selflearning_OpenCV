import cv2
import numpy as np
import mediapipe as mp
import time

cam = cv2.VideoCapture(1)
mpHands = mp.solutions.hands
hands = mpHands.Hands()
mpDraw = mp.solutions.drawing_utils
handLmsStyle = mpDraw.DrawingSpec(color=(0,0,255), thickness=5)
handConStyle = mpDraw.DrawingSpec(color=(0,255,0), thickness=10)

pTime = 0
cTime = 0

while True:
    ret, img = cam.read()
    img = cv2.resize(img,None,fx=2,fy=2)
    if ret:
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        result = hands.process(imgRGB)
        # print(result.multi_hand_landmarks)
        imgHeight = img.shape[0]
        imgWidth = img.shape[1]

        if result.multi_hand_landmarks:
            for handLms in result.multi_hand_landmarks:
                mpDraw.draw_landmarks(img,handLms,mpHands.HAND_CONNECTIONS,handLmsStyle,handConStyle)
                for i,lm in enumerate(handLms.landmark):
                    xPos = int(lm.x*imgWidth)
                    yPos = int(lm.y*imgHeight)
                    # cv2.putText(img,str(i),(xPos-25, yPos+5),cv2.FONT_HERSHEY_COMPLEX_SMALL,1,(1,0,255),2)
                    if i == 4:
                        cv2.circle(img,(xPos,yPos),20, (200,100,255),cv2.FILLED)
                    print(i,lm.x,lm.y)

        cTime = time.time()
        fps = 1/(cTime-pTime)
        pTime = cTime
        cv2.putText(img, f"FPS:{int(fps)}", (30,50), cv2.FONT_HERSHEY_COMPLEX_SMALL,2,(255,0,0),2)

        cv2.imshow('img',img)
    if cv2.waitKey(1)==ord('q'):
        break