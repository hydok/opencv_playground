import cv2

capture = cv2.VideoCapture(0)
capture.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

while cv2.waitKey(33) < 0:
    ret, frame = capture.read()
    if not ret:
        break

    # 1: 좌우 반전 (거울 모드)
#    frame = cv2.flip(frame, 1)

    cv2.imshow("VideoFrame", frame)

capture.release()
cv2.destroyAllWindows()
