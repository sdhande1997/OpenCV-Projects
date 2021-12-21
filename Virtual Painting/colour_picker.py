import cv2
import numpy as np


def nothing(x):
    print()


cap = cv2.VideoCapture(0)
cv2.namedWindow("Tracking")
cv2.createTrackbar("Hue min", "Tracking", 0, 180, nothing)
cv2.createTrackbar("Hue max", "Tracking", 180, 180, nothing)
cv2.createTrackbar("Sat min", "Tracking", 0, 255, nothing)
cv2.createTrackbar("Sat max", "Tracking", 255, 255, nothing)
cv2.createTrackbar("Val min", "Tracking", 0, 255, nothing)
cv2.createTrackbar("Val max", "Tracking", 255, 255, nothing)

while cap.isOpened:
    ret, frame = cap.read()
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    h_min = cv2.getTrackbarPos("Hue min", "Tracking")
    h_max = cv2.getTrackbarPos("Hue max", "Tracking")
    s_min = cv2.getTrackbarPos("Sat min", "Tracking")
    s_max = cv2.getTrackbarPos("Sat max", "Tracking")
    v_min = cv2.getTrackbarPos("Val min", "Tracking")
    v_max = cv2.getTrackbarPos("Val max", "Tracking")
    lower = np.array([h_min, s_min, v_min])
    upper = np.array([h_max, s_max, v_max])
    mask = cv2.inRange(hsv, lower, upper)
    result = cv2.bitwise_and(frame, frame, mask=mask)

    cv2.imshow("frame", frame)
    cv2.imshow("mask", mask)
    cv2.imshow("result", result)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()




