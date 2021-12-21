import cv2
import numpy as np


frameWidth = 400
frameHeight = 600
cap = cv2.VideoCapture(0)
cap.set(3, frameWidth)
cap.set(4, frameHeight)
cap.set(10, 150)
print(cap.get(3))
print(cap.get(4))


def preProcessing(frame):
    imGray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    imBlur = cv2.GaussianBlur(imGray, (5,5), 1)
    imCanny = cv2.Canny(imBlur, 65, 0)
    kernel = np.ones((5,5))
    dilate = cv2.dilate(imCanny, kernel, iterations = 2)
    imThresh = cv2.erode(dilate, kernel, iterations = 1)

    return imThresh


def getContour(frame):
    contours, hierarchy = cv2.findContours(frame, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    biggest = np.array([])
    maxArea = 0
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > 5000:
            # cv2.drawContours(frameContour, cnt, -1, (255,0,0), 2)
            peri = cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
            if area > maxArea and len(approx) == 4:
                biggest = approx
                maxArea = area
    cv2.drawContours(frameContour, biggest, -1, (255, 0, 0), 20)
    return biggest


def reorder(myPoints):
    myPoints = myPoints.reshape((4,2))
    mynewPoints = np.zeros((4,1,2), np.uint32)
    add = myPoints.sum(1)
    #print("add", add)

    mynewPoints[0] = myPoints[np.argmin(add)]    # argmin/argmax pass on just the index of min/max values in list
    mynewPoints[3] = myPoints[np.argmax(add)]
    diff = np.diff(myPoints, axis = 1)
    mynewPoints[1] = myPoints[np.argmin(diff)]
    mynewPoints[2] = myPoints[np.argmax(diff)]
    #print("Diff", diff)
    #print("My New Points", mynewPoints)
    return mynewPoints


def getWarp(frame, biggest):
    biggest = reorder(biggest)
    print(biggest)
    pts1 = np.float32(biggest)
    pts2 = np.float32([[0, 0], [frameWidth, 0], [0, frameHeight], [frameWidth, frameHeight]])
    matrix = cv2.getPerspectiveTransform(pts1, pts2)
    frameOutput = cv2.warpPerspective(frame, matrix, (frameWidth, frameHeight))

    frameCropped = frameOutput[20:frameOutput.shape[0]-20, 20:frameOutput.shape[1]-20]
    frameCropped = cv2.resize(frameCropped, (frameWidth, frameHeight))

    return frameCropped


def stackImages(scale,imgArray):
    rows = len(imgArray)
    cols = len(imgArray[0])
    rowsAvailable = isinstance(imgArray[0], list)
    width = imgArray[0][0].shape[1]
    height = imgArray[0][0].shape[0]
    if rowsAvailable:
        for x in range ( 0, rows):
            for y in range(0, cols):
                if imgArray[x][y].shape[:2] == imgArray[0][0].shape [:2]:
                    imgArray[x][y] = cv2.resize(imgArray[x][y], (0, 0), None, scale, scale)
                else:
                    imgArray[x][y] = cv2.resize(imgArray[x][y], (imgArray[0][0].shape[1], imgArray[0][0].shape[0]), None, scale, scale)
                if len(imgArray[x][y].shape) == 2: imgArray[x][y]= cv2.cvtColor( imgArray[x][y], cv2.COLOR_GRAY2BGR)
        imageBlank = np.zeros((height, width, 3), np.uint8)
        hor = [imageBlank]*rows
        hor_con = [imageBlank]*rows
        for x in range(0, rows):
            hor[x] = np.hstack(imgArray[x])
        ver = np.vstack(hor)
    else:
        for x in range(0, rows):
            if imgArray[x].shape[:2] == imgArray[0].shape[:2]:
                imgArray[x] = cv2.resize(imgArray[x], (0, 0), None, scale, scale)
            else:
                imgArray[x] = cv2.resize(imgArray[x], (imgArray[0].shape[1], imgArray[0].shape[0]), None,scale, scale)
            if len(imgArray[x].shape) == 2: imgArray[x] = cv2.cvtColor(imgArray[x], cv2.COLOR_GRAY2BGR)
        hor= np.hstack(imgArray)
        ver = hor
    return ver

while True:
    ret, frame = cap.read()
    frame = cv2.resize(frame, (frameWidth, frameHeight))

    frameContour = frame.copy()
    imThresh = preProcessing(frame)
    biggest = getContour(imThresh)
    if biggest.size != 0:
        frameWarped = getWarp(frame, biggest)
        frameArray = ([frame, imThresh],
                       [frameContour, frameWarped])
    else:
        frameArray = ([frame, imThresh],
                      [frame, frame])
    stackedImages = stackImages(0.6, frameArray)

    cv2.imshow("Warp", stackedImages)
    cv2.imshow("Warp Image", frameWarped)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cv2.destroyAllWindows()