import cv2
import numpy as np

img1 = cv2.imread("Image Queue/Hitchhikers Guide To The Galaxy_All Parts.jpg")
img2 = cv2.imread("Image Train/HGTG1.jpg")
img1 = cv2.resize(img1, (370, 540))
img2 = cv2.resize(img2, (400, 512))

orb = cv2.ORB_create(nfeatures=1000)

kp1, des1 = orb.detectAndCompute(img1, None)
kp2, des2 = orb.detectAndCompute(img2, None)
# imgkp1 = cv2.drawKeypoints(img1, kp1, None)
# imgkp2 = cv2.drawKeypoints(img2, kp2, None)

bf = cv2.BFMatcher()
matches = bf.knnMatch(des1, des2, k=2)
good = []
for m,n in matches:
    if m.distance < 0.80*n.distance:
        good.append([m])

print(len(good))
img3 = cv2.drawMatchesKnn(img1, kp1, img2, kp2, good, None, flags=2)

cv2.imshow("Image1", img3)
#cv2.imshow("Image2", imgkp2)
cv2.waitKey(0)
cv2.destroyAllWindows()