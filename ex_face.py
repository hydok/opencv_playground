import os
import sys
import urllib.request
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

# -------------------------------------------------------------
# 모델 파일 설정 및 자동 다운로드
# -------------------------------------------------------------
YUNET_MODEL = "face_detection_yunet_2023mar.onnx"
YUNET_URL = "https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx"

FER_MODEL = "facial_expression_recognition_mobilefacenet_2022july.onnx"
FER_URL = "https://github.com/opencv/opencv_zoo/raw/main/models/facial_expression_recognition/facial_expression_recognition_mobilefacenet_2022july.onnx"

# 표준 5개 랜드마크 기준 좌표 (얼굴 정렬용)
STD_POINTS = np.array(
    [
        [38.2946, 51.6963],  # 오른쪽 눈
        [73.5318, 51.5014],  # 왼쪽 눈
        [56.0252, 71.7366],  # 코 끝
        [41.5493, 92.3655],  # 오른쪽 입꼬리
        [70.7299, 92.2041],  # 왼쪽 입꼬리
    ],
    dtype=np.float32,
)

# 7가지 표정 레이블, 이모지 문자 및 대표 색상 (BGR)
EMOTIONS = ["Angry", "Disgust", "Fearful", "Happy", "Neutral", "Sad", "Surprised"]

EMOJI_CHARS = {
    "Angry": "😡",
    "Disgust": "🤢",
    "Fearful": "😨",
    "Happy": "😄",
    "Neutral": "😐",
    "Sad": "😢",
    "Surprised": "😲",
}

EMOTION_COLORS = {
    "Angry": (0, 0, 255),       # 빨간색
    "Disgust": (0, 140, 255),    # 주황색
    "Fearful": (128, 0, 128),    # 보라색
    "Happy": (0, 255, 0),        # 초록색
    "Neutral": (200, 200, 200),  # 회색
    "Sad": (255, 100, 0),        # 파란색 계열
    "Surprised": (0, 255, 255),  # 노란색
}


def download_model_if_missing(file_path: str, url: str):
    """모델 파일이 없으면 OpenCV Zoo 저장소에서 자동으로 다운로드합니다."""
    if not os.path.exists(file_path):
        print(f"모델 다운로드 중: {file_path} ...")
        urllib.request.urlretrieve(url, file_path)
        print("다운로드 완료!")


def init_emoji_icons(size: int = 40):
    """
    macOS의 Apple Color Emoji 폰트를 사용하여 표정별 이모지 아이콘(BGRA)을 사전 생성합니다.
    (Apple Color Emoji는 20, 32, 40, 48, 64, 96 크기 등을 지원합니다)
    """
    font_candidates = [
        "/System/Library/Fonts/Apple Color Emoji.ttc",
        "/System/Library/Fonts/Supplemental/AppleColorEmoji.ttc",
    ]
    font_path = next((p for p in font_candidates if os.path.exists(p)), None)
    if font_path is None:
        return {}

    try:
        font = ImageFont.truetype(font_path, size)
        icons = {}
        for name, char in EMOJI_CHARS.items():
            img = Image.new("RGBA", (size, size), (0, 0, 0, 0))
            draw = ImageDraw.Draw(img)
            draw.text((0, 0), char, font=font, embedded_color=True)
            arr = np.array(img)
            icons[name] = cv2.cvtColor(arr, cv2.COLOR_RGBA2BGRA)
        return icons
    except Exception as e:
        print(f"이모지 폰트 로드 실패 ({e}), 텍스트로만 표시합니다.")
        return {}


def overlay_rgba(background, overlay, x, y):
    """알파 채널이 포함된 BGRA 이미지를 BGR 배경 이미지 위에 자연스럽게 합성합니다."""
    h, w = background.shape[:2]
    oh, ow = overlay.shape[:2]

    # 화면 경계 클리핑
    x1, y1 = max(x, 0), max(y, 0)
    x2, y2 = min(x + ow, w), min(y + oh, h)

    if x1 >= x2 or y1 >= y2:
        return

    ox1, oy1 = x1 - x, y1 - y
    ox2, oy2 = ox1 + (x2 - x1), oy1 + (y2 - y1)

    crop_overlay = overlay[oy1:oy2, ox1:ox2]
    alpha = crop_overlay[:, :, 3] / 255.0

    for c in range(3):
        background[y1:y2, x1:x2, c] = (
            (1.0 - alpha) * background[y1:y2, x1:x2, c]
            + alpha * crop_overlay[:, :, c]
        )


def main():
    # 모델 파일 존재 확인
    download_model_if_missing(YUNET_MODEL, YUNET_URL)
    download_model_if_missing(FER_MODEL, FER_URL)

    # 1. 이모지 아이콘 사전 렌더링 (얼굴 라벨용 40px, 화면 우상단 배지용 64px)
    emoji_icons_40 = init_emoji_icons(size=40)
    emoji_icons_64 = init_emoji_icons(size=64)

    # 2. 카메라 열기
    capture = cv2.VideoCapture(0)
    capture.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    if not capture.isOpened():
        print("카메라를 열 수 없습니다. 카메라 연결 및 macOS 권한을 확인해주세요.")
        sys.exit()

    frame_width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # 3. 얼굴 감지 모델(YuNet) 초기화
    detector = cv2.FaceDetectorYN_create(
        model=YUNET_MODEL,
        config="",
        input_size=(frame_width, frame_height),
        score_threshold=0.6,
        nms_threshold=0.3,
        top_k=5000,
    )

    # 4. 표정 인식 모델(MobileFaceNet FER) 로드
    fer_net = cv2.dnn.readNet(FER_MODEL)

    print("==================================================")
    print("실시간 얼굴 표정 & 이모지 인식 시작 (종료: 'q' 키 또는 ESC)")
    print("==================================================")

    while True:
        ret, frame = capture.read()
        if not ret:
            print("카메라 프레임을 읽을 수 없습니다.")
            break

        # 거울 모드 (좌우 반전)
        frame = cv2.flip(frame, 1)

        # 해상도 변경 시 detector 입력 크기 동기화
        h, w = frame.shape[:2]
        detector.setInputSize((w, h))

        # 얼굴 감지 수행
        _, faces = detector.detect(frame)

        top_emotion = None
        top_confidence = 0

        if faces is not None:
            for face in faces:
                # 바운딩 박스: [x, y, width, height]
                bbox = face[:4].astype(int)
                x, y, bw, bh = bbox

                # 5개 랜드마크 추출 (오른쪽눈, 왼쪽눈, 코, 오른쪽입꼬리, 왼쪽입꼬리)
                landmarks = face[4:14].reshape((5, 2)).astype(np.float32)

                # 아핀 변환 행렬을 이용해 얼굴 정렬 (112x112)
                M, _ = cv2.estimateAffinePartial2D(landmarks, STD_POINTS)
                if M is None:
                    continue

                aligned_face = cv2.warpAffine(frame, M, (112, 112))

                # 표정 인식 모델 추론
                blob = cv2.dnn.blobFromImage(
                    aligned_face,
                    scalefactor=1.0 / 127.5,
                    size=(112, 112),
                    mean=(127.5, 127.5, 127.5),
                    swapRB=True,
                )
                fer_net.setInput(blob)
                output = fer_net.forward()[0]

                # Softmax 확률 계산
                exp_scores = np.exp(output - np.max(output))
                probs = exp_scores / np.sum(exp_scores)

                pred_idx = np.argmax(probs)
                emotion = EMOTIONS[pred_idx]
                confidence = probs[pred_idx] * 100

                if confidence > top_confidence:
                    top_emotion = emotion
                    top_confidence = confidence

                color = EMOTION_COLORS.get(emotion, (0, 255, 0))

                # 얼굴 바운딩 박스 그리기
                cv2.rectangle(frame, (x, y), (x + bw, y + bh), color, 2)

                # 5개 랜드마크 포인트 점 표시
                for lx, ly in landmarks:
                    cv2.circle(frame, (int(lx), int(ly)), 2, (0, 255, 255), -1)

                # -------------------------------------------------------------
                # 이모지 + 텍스트 라벨 배너 그리기
                # -------------------------------------------------------------
                label_text = f"{emotion} ({confidence:.0f}%)"
                (text_w, text_h), baseline = cv2.getTextSize(
                    label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2
                )

                icon = emoji_icons_40.get(emotion)
                icon_w = icon.shape[1] if icon is not None else 0
                icon_h = icon.shape[0] if icon is not None else 0

                banner_h = max(icon_h, text_h) + 12
                banner_w = icon_w + text_w + 20

                # 얼굴 위쪽에 배너 배치 (상단 공간 부족 시 얼굴 아래쪽에 배치)
                banner_x = x
                banner_y = (y - banner_h - 6) if (y >= banner_h + 6) else (y + bh + 6)

                # 배너 배경 박스
                cv2.rectangle(
                    frame,
                    (banner_x, banner_y),
                    (banner_x + banner_w, banner_y + banner_h),
                    color,
                    cv2.FILLED,
                )

                # 이모지 아이콘 합성
                if icon is not None:
                    icon_y = banner_y + (banner_h - icon_h) // 2
                    overlay_rgba(frame, icon, banner_x + 5, icon_y)

                # 텍스트 라벨 출력 (이모지 오른쪽)
                text_x = banner_x + icon_w + 10 if icon is not None else banner_x + 8
                text_y = banner_y + banner_h - 12
                cv2.putText(
                    frame,
                    label_text,
                    (text_x, text_y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 0, 0),
                    2,
                    cv2.LINE_AA,
                )

        # -------------------------------------------------------------
        # 화면 우측 상단 대형 이모지 배지 표시 (얼굴 감지 시)
        # -------------------------------------------------------------
        if top_emotion and top_emotion in emoji_icons_64:
            big_icon = emoji_icons_64[top_emotion]
            badge_x = w - 85
            badge_y = 15
            # 반투명 어두운 배경 박스
            overlay_bg = frame[badge_y - 5 : badge_y + 75, badge_x - 5 : badge_x + 75]
            if overlay_bg.shape[0] == 80 and overlay_bg.shape[1] == 80:
                black_box = np.zeros_like(overlay_bg)
                cv2.addWeighted(overlay_bg, 0.4, black_box, 0.6, 0, overlay_bg)
                frame[badge_y - 5 : badge_y + 75, badge_x - 5 : badge_x + 75] = overlay_bg
            overlay_rgba(frame, big_icon, badge_x, badge_y)

        # 상단 안내 텍스트 표시
        cv2.putText(
            frame,
            "Press 'q' or ESC to exit",
            (15, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )

        cv2.imshow("Facial Expression Recognition", frame)

        # 'q' 또는 ESC 키 입력 시 종료
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q") or key == 27:
            break

    capture.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
