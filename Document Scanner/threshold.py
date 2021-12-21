import cv2

cap = cv2.VideoCapture(1)
cv2.namedWindow("Tracking")


def nothing(x):
    print()


cv2.createTrackbar("Threshold 1", "Tracking", 0, 255, nothing)
cv2.createTrackbar("Threshold 2", "Tracking", 0, 255, nothing)


while cap.isOpened():
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5,5), 1)
    thresh1 = cv2.getTrackbarPos("Threshold 1", "Tracking")
    thresh2 = cv2.getTrackbarPos("Threshold 2", "Tracking")
    canny = cv2.Canny(blur, thresh1, thresh2)
    cv2.imshow("Canny", canny)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break


cap.release()
cv2.destroyAllWindows()