import sys
import numpy as np
import cv2


def resize_image(img, target_width=480):
    """화면에 한눈에 들어오도록 비율을 유지하며 리사이즈합니다."""
    h, w = img.shape[:2]
    target_height = int(h * (target_width / w))
    return cv2.resize(img, (target_width, target_height))


def get_gray_hist_image(img, width=480, height=200, title=''):
    """그레이스케일 영상의 히스토그램 이미지를 생성합니다."""
    hist = cv2.calcHist([img], [0], None, [256], [0, 256]).flatten()
    hist_img = np.zeros((height, width), dtype=np.uint8)
    cv2.normalize(hist, hist, 0, height - 20, cv2.NORM_MINMAX)

    bin_w = width / 256
    for i in range(256):
        x1 = int(i * bin_w)
        x2 = int((i + 1) * bin_w)
        y = int(hist[i])
        cv2.rectangle(hist_img, (x1, height), (x2, height - y), 255, -1)

    if title:
        cv2.putText(hist_img, title, (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, 255, 1)
    return hist_img


def get_color_hist_image(img, width=480, height=200, title=''):
    """컬러 영상의 B, G, R 채널별 히스토그램 이미지를 생성합니다."""
    hist_img = np.zeros((height, width, 3), dtype=np.uint8)
    colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255)]  # BGR
    bin_w = width / 256

    for i, col in enumerate(colors):
        hist = cv2.calcHist([img], [i], None, [256], [0, 256]).flatten()
        cv2.normalize(hist, hist, 0, height - 20, cv2.NORM_MINMAX)
        for x in range(255):
            pt1 = (int(x * bin_w), height - int(hist[x]))
            pt2 = (int((x + 1) * bin_w), height - int(hist[x + 1]))
            cv2.line(hist_img, pt1, pt2, col, 2)

    if title:
        cv2.putText(hist_img, title, (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)
    return hist_img


# -------------------------------------------------------------
# 1. 그레이스케일 영상: 밝기 조절 전/후 이미지 및 히스토그램 출력
# -------------------------------------------------------------
src = cv2.imread('images/home.jpg', cv2.IMREAD_GRAYSCALE)

if src is None:
    print('Image load failed!')
    sys.exit()

dst = cv2.add(src, 50)

# 화면에 맞게 이미지 리사이즈 및 텍스트 표시
src_disp = resize_image(src, target_width=480)
dst_disp = resize_image(dst, target_width=480)
cv2.putText(src_disp, 'Before (src)', (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, 255, 2)
cv2.putText(dst_disp, 'After (+50)', (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, 255, 2)

# 이미지 좌우 결합 (상단)
img_row = np.hstack([src_disp, dst_disp])

# 히스토그램 좌우 결합 (하단)
hist_src = get_gray_hist_image(src, width=480, height=200, title='Before Hist')
hist_dst = get_gray_hist_image(dst, width=480, height=200, title='After Hist')
hist_row = np.hstack([hist_src, hist_dst])

# 상단(이미지)과 하단(히스토그램)을 수직 결합하여 한 창에 표시
total_gray = np.vstack([img_row, hist_row])

cv2.imshow('Grayscale: Images & Histograms (Before vs After)', total_gray)
cv2.waitKey()
cv2.destroyAllWindows()


# -------------------------------------------------------------
# 2. 컬러 영상: 밝기 조절 전/후 이미지 및 히스토그램 출력
# -------------------------------------------------------------
src = cv2.imread('images/home.jpg')

if src is None:
    print('Image load failed!')
    sys.exit()

dst = cv2.add(src, (50, 50, 50, 0))

# 화면에 맞게 이미지 리사이즈 및 텍스트 표시
src_disp = resize_image(src, target_width=480)
dst_disp = resize_image(dst, target_width=480)
cv2.putText(src_disp, 'Before (src)', (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
cv2.putText(dst_disp, 'After (+50)', (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

# 컬러 이미지 좌우 결합 (상단)
img_row_col = np.hstack([src_disp, dst_disp])

# 컬러 히스토그램 좌우 결합 (하단)
hist_src_col = get_color_hist_image(src, width=480, height=200, title='Before Color Hist')
hist_dst_col = get_color_hist_image(dst, width=480, height=200, title='After Color Hist')
hist_row_col = np.hstack([hist_src_col, hist_dst_col])

# 상단(컬러 이미지)과 하단(컬러 히스토그램)을 수직 결합하여 한 창에 표시
total_color = np.vstack([img_row_col, hist_row_col])

cv2.imshow('Color: Images & Histograms (Before vs After)', total_color)
cv2.waitKey()
cv2.destroyAllWindows()


