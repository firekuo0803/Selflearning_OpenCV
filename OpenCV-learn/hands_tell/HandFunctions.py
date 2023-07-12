import cv2
import numpy as np
import mediapipe as mp
import time

class handDetector():
    def __init__(self, mode=False, maxHands =2, detectionCon =0.5, trackCon = 0.5):
        self.mode = mode
        self.maxHands = maxHands
        self.detectionCon = detectionCon
        self.trackCon = trackCon

        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(self.mode,self.maxHands,self.detectionCon,self.trackCon)
        self.mpDraw = mp.solutions.drawing_utils

        self.handLmsStyle = self.mpDraw.DrawingSpec(color=(0, 0, 255), thickness=5)
        self.handConStyle = self.mpDraw.DrawingSpec(color=(0, 255, 0), thickness=10)

    def findHands(self, img, draw = True):
        # if ret:
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.result = self.hands.process(imgRGB)
        # print(result.multi_hand_landmarks)
        if self.result.multi_hand_landmarks:
            for handLms in self.result.multi_hand_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(img, handLms, self.mpHands.HAND_CONNECTIONS, self.handLmsStyle, self.handConStyle)
        return img

    def findPosition(self, img, handNum = 0, All = False, number = False,
                     DotX = -1, bs = 15, col = (200, 100, 255)):
        lmList = []

        imgHeight = img.shape[0]
        imgWidth = img.shape[1]
        if self.result.multi_hand_landmarks:
            myHand = self.result.multi_hand_landmarks[handNum]

            for i, lm in enumerate(myHand.landmark):
                xPos = int(lm.x * imgWidth)
                yPos = int(lm.y * imgHeight)
                print(i, lm.x, lm.y)

                lmList.append([i,xPos,yPos])
                if All:
                    cv2.circle(img, (xPos, yPos), bs, col, cv2.FILLED)

                if i == DotX:
                    cv2.circle(img, (xPos, yPos), bs, col, cv2.FILLED)

                if number:
                    cv2.putText(img,str(i),(xPos-25, yPos+5),cv2.FONT_HERSHEY_COMPLEX_SMALL,1,(1,0,255),2)

        return lmList



def main():
    pTime = 0
    cTime = 0
    cam = cv2.VideoCapture(1)
    detector = handDetector()
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




if __name__ == "__main__":
    main()
