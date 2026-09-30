import cv2

image = cv2.imread("images/moon.jpg", cv2.IMREAD_ANYCOLOR)
dst = cv2.resize(image, dsize=(0, 0), fx=0.5, fy=0.5, interpolation=cv2.INTER_LINEAR)

cv2.imshow("Moon", dst)

cv2.waitKey()
cv2.destroyAllWindows()