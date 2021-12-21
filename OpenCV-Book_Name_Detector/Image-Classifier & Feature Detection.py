import cv2
import numpy as np
import os

path = "Image Queue"
orb = cv2.ORB_create(nfeatures=1000)
##### Image Import #####
images = []
classnames = []
myList = os.listdir(path)
print("MyList = ", + len(myList))
for cl in myList:
    imgCur = cv2.imread(f"{path}/{cl}", 0)
    images.append(imgCur)
    classnames.append(os.path.splitext(cl)[0])


def findDes(images):
    desList = []
    for img in images:
        kp1, des1 = orb.detectAndCompute(img, None)
        desList.append(des1)
    return desList


desList = findDes(images)
print("1stdesList = ", + len(desList))


def findID(frame, desList, thresh = 15):
    kp2, des2 = orb.detectAndCompute(frame, None)
    bf = cv2.BFMatcher()
    matchList = []
    finalVal = -1
    for des in desList:
            matches = bf.knnMatch(des, des2, k=2)
            good = []
            for m, n in matches:
                if m.distance < 0.80*n.distance:
                    good.append([m])
            matchList.append(len(good))
    print(matchList)
    if len(matchList) !=0:
        if max(matchList) > thresh:
            finalVal = matchList.index(max(matchList))
    return finalVal


cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    OriginalFrame = frame.copy()
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    id = findID(frame, desList)
    if id!= -1:
        cv2.putText(OriginalFrame, classnames[id], (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1,(0,0,255), 2)
    cv2.imshow("frame", OriginalFrame)
    if cv2.waitKey(1) == ord("q"):
        break


