import sys
import cv2

# 이미지 로드
src = cv2.imread('images/2.jpg')

if src is None:
    print('Image load failed!')
    sys.exit()

window_name = 'Brightness Control'
cv2.namedWindow(window_name)


def on_trackbar(pos):
    # 트랙바 범위: 0 ~ 200 (기본값 100 기준 -100 ~ +100 조절)
    offset = pos - 100
    dst = cv2.convertScaleAbs(src, alpha=1.0, beta=offset)
    cv2.imshow(window_name, dst)


# 트랙바 생성: 트랙바 이름, 윈도우 이름, 초기값, 최댓값, 콜백 함수
cv2.createTrackbar('Brightness', window_name, 100, 200, on_trackbar)

# 초기 원본 이미지 표시
on_trackbar(100)

cv2.waitKey(0)
cv2.destroyAllWindows()
