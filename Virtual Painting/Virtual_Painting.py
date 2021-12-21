import cv2
import numpy as np

frameWidth = 720
frameHeight = 1000
cap = cv2.VideoCapture(0)
cap.set(3, frameWidth)
cap.set(4, frameHeight)
cap.set(10, 130)

myColors = [[103, 96, 0, 133, 255, 255],     # Blue
            [15, 171, 144, 180, 255, 255]]       # Yellow
            # [103, 93, 0, 139, 255, 255]]

myColorValues = [[255, 0, 0],  # in BGR format values
                 [0, 255, 255]]
                 # [255, 0, 0]]

myPoints = []       # [ x , y , colorID ]


def findColor(frame, myColors, myColorValues):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    count = 0
    newPoints = []
    for color in myColors:
        lower = np.array(color[0:3])
        upper = np.array(color[3:6])
        mask = cv2.inRange(hsv, lower, upper)
        x, y = getContours(mask)
        cv2.circle(frameResult, (x, y), 10, myColorValues[count], cv2.FILLED)
        if x != 0 and y != 0:
            newPoints.append([x, y, count])
        count += 1
        # cv2.imshow(str(color[0]), mask)
    return newPoints


def drawOnCanvas(myPoints, myColorValues):
     for point in myPoints:
         cv2.circle(frameResult, (point[0], point[1]), 10, myColorValues[point[2]], cv2.FILLED)


def getContours(frame):
    contours, hierarchy = cv2.findContours(frame, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    x, y, w, h = 0, 0, 0, 0
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > 500:
            cv2.drawContours(frameResult, cnt, -1, (255, 0, 0), 3)
            peri = cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
            x, y, w, h = cv2.boundingRect(approx)
    return x + w // 2, y


while True:
    ret, frame = cap.read()
    frameResult = frame.copy()
    newPoints = findColor(frame, myColors, myColorValues)
    if len(newPoints)!=0:
        for newP in newPoints:
            myPoints.append(newP)
    if len(myPoints)!=0:
        drawOnCanvas(myPoints, myColorValues)
    cv2.imshow("Frame", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break
