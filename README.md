# 🖼️ OpenCV Playground

OpenCV와 Python을 활용하여 컴퓨터 비전(Computer Vision)의 핵심 기초 기능(영상 입출력, 픽셀 조작, 색 공간 변환, 히스토그램 평활화, 투시 변환, 크로마키 합성)부터 딥러닝 기반 얼굴/표정 분석(ONNX) 및 생성형 AI(Gemini 3.8 Flash Function Calling) 연동까지 실습할 수 있는 프로젝트입니다.

---

## 📌 목차
1. [개발 및 실행 환경](#-개발-및-실행-환경)
2. [프로젝트 구조](#-프로젝트-구조)
3. [구현 코드 및 컴퓨터 비전 개념 정리](#-구현-코드-및-컴퓨터-비전-개념-정리)
   - [1. 기초 영상 입출력 및 조작](#1-기초-영상-입출력-및-조작)
   - [2. 밝기 및 명암비 조절](#2-밝기-및-명암비-조절)
   - [3. 히스토그램 및 색 공간 분할](#3-히스토그램-및-색-공간-분할)
   - [4. 영상 합성 및 기하학적 변환](#4-영상-합성-및-기하학적-변환)
   - [5. 딥러닝 기반 얼굴 및 표정 인식](#5-딥러닝-기반-얼굴-및-표정-인식)
   - [6. 생성형 AI (Gemini) Function Calling 연동](#6-생성형-ai-gemini-function-calling-연동)
   - [7. 유틸리티](#7-유틸리티)
4. [핵심 컴퓨터 비전 개념 요약](#-핵심-컴퓨터-비전-개념-요약)

---

## ⚙️ 개발 및 실행 환경

- **OS**: macOS
- **Python 가상환경**: 3.14
- **주요 라이브러리**:
  - `opencv-python` (5.0.0+)
  - `numpy`
  - `pillow`
  - `google-genai` (2.25.0+)
 

### 환경 변수 설정 (Gemini 연동 시 필요)
```bash
export GEMINI_API_KEY="your-gemini-api-key"
```

---

## 📁 프로젝트 구조

```text
opencv_playground/
├── images/                           # 실습용 샘플 이미지 (2.jpg, home.jpg, candies.png, road.png 등)
├── videos/                           # 크로마키 합성용 동영상 (woman.mp4, raining.mp4)
│
├── ex_imread.py                      # 영상 파일 읽기, 크기 조절 및 화면 출력
├── ex_crop.py                        # NumPy 슬라이싱을 이용한 관심영역(ROI) 크롭
├── ex_resize.py                      # 피라미드 업/다운샘플링 (pyrUp, pyrDown)
├── ex_imcamera.py                    # 웹캠 실시간 영상 스트리밍
│
├── ex_brightness.py                  # 밝기 조절(cv2.add) 및 전/후 히스토그램 시각화
├── ex_brightness_trackbar.py         # 트랙바 기반 실시간 밝기 조절 대화형 GUI
├── ex_equalize.py                    # 그레이스케일 및 컬러(YCrCb) 히스토그램 평활화
│
├── ex_histogram.py                   # 그레이스케일 히스토그램 계산 및 직접 그리기
├── ex_inrange2.py                    # HSV 색 공간 변환 및 특정 색상 영역 마스킹
│
├── ex_chroma_key.py                  # HSV 기반 실시간 동영상 크로마키 배경 합성
├── ex_docuscan.py                    # 마우스 드래그 4점 지정 및 원근 투시 변환 (문서 스캐너)
│
├── ex_face.py                        # YuNet 얼굴 검출 + MobileFaceNet 표정 인식 + 이모지 오버레이
├── ex_gemini_brightness.py           # Gemini 3.8 Flash의 Function Calling을 통한 지능형 밝기 제어
│
├── convert_webp_to_jpg.py            # WebP -> JPG 포맷 일괄 변환 유틸리티
├── .gitignore                        # Git 추적 제외 설정
└── README.md                         # 프로젝트 매뉴얼 및 개념 정리 문서
```

---

## 📚 구현 코드 및 컴퓨터 비전 개념 정리

### 1. 기초 영상 입출력 및 조작

#### [`ex_imread.py`](ex_imread.py) - 영상 불러오기 및 리사이즈
- **기능**: 이미지 파일을 불러와 비율을 50%로 축소한 뒤 창에 표시합니다.
- **핵심 개념**:
  - `cv2.imread(filename, flags)`: BGR(기본값) 또는 그레이스케일 형태로 픽셀 배열을 메모리에 로드합니다.
  - `cv2.resize()`: `fx`, `fy` 인수를 통해 비율 기반 리사이즈 및 선형 보간법(`INTER_LINEAR`)을 적용합니다.

#### [`ex_crop.py`](ex_crop.py) - 관심 영역(ROI) 추출
- **기능**: 이미지의 특정 좌표 영역만 슬라이싱하여 추출합니다.
- **핵심 개념**:
  - OpenCV 영상은 NumPy `ndarray` (`[H, W, C]`) 구조입니다. `img[y1:y2, x1:x2]` 형태로 원하는 영역(ROI)을 잘라냅니다.
  - `.copy()`를 호출하여 원본과의 메모리 공유를 방지하고 독립적인 버퍼를 확보합니다.

#### [`ex_resize.py`](ex_resize.py) - 영상 피라미드
- **기능**: 영상의 해상도를 2배로 확대하거나 1/2로 축소합니다.
- **핵심 개념**:
  - `cv2.pyrDown()`: 가우시안 블러링을 거친 후 짝수 행/열을 제거하여 에일리어싱(Aliasing) 없이 크기를 절반으로 줄입니다.
  - `cv2.pyrUp()`: 0을 삽입한 후 가우시안 필터로 보간하여 2배 크기로 확대합니다.

#### [`ex_imcamera.py`](ex_imcamera.py) - 실시간 카메라 스트리밍
- **기능**: 기본 웹캠(디바이스 0)으로부터 실시간 프레임을 수신하여 화면에 표시합니다.
- **핵심 개념**:
  - `cv2.VideoCapture(0)`: 카메라 디바이스 오픈 및 FPS/해상도 설정.
  - `cv2.waitKey(delay)`: 프레임 간격을 조절하고 키 입력을 대기합니다 (33ms = 약 30FPS).

---

### 2. 밝기 및 명암비 조절

#### [`ex_brightness.py`](ex_brightness.py) - 밝기 조절과 히스토그램 분석
- **기능**: 그레이스케일 및 BGR 컬러 영상의 밝기를 높이고, 전/후 픽셀 분포(히스토그램)를 비교 시각화합니다.
- **핵심 개념**:
  - **포화 연산 (Saturation Arithmetic)**: 일반 덧셈(`+`)을 사용하면 `uint8` 오버플로우가 발생하여 255를 초과할 때 값이 0으로 돌아가 색 반전이 일어납니다. `cv2.add()`는 값이 255를 초과하면 자동으로 255로 제한(클리핑)합니다.
  - **컬러 밝기 조절**: BGR 영상에서는 각 채널에 동일한 스칼라를 더해야 색상 왜곡 없이 밝기만 증가합니다 (`(50, 50, 50, 0)`).

#### [`ex_brightness_trackbar.py`](ex_brightness_trackbar.py) - 트랙바 대화형 밝기 조절
- **기능**: 마우스로 조작 가능한 슬라이더를 통해 실시간으로 -100 ~ +100 범위의 밝기를 조절합니다.
- **핵심 개념**:
  - `cv2.createTrackbar()`: 윈도우 상단에 슬라이더 GUI 컴포넌트를 부착하고 콜백 함수를 연결합니다.
  - `cv2.convertScaleAbs(src, alpha=1.0, beta=offset)`: 스케일링($\alpha$) 및 오프셋($\beta$)을 적용한 뒤 절댓값 및 uint8 포화 연산을 가장 효율적으로 수행하는 OpenCV 표준 함수입니다.

#### [`ex_equalize.py`](ex_equalize.py) - 히스토그램 평활화 (Histogram Equalization)
- **기능**: 명암 대비가 낮거나 어두운 영상의 픽셀 분포를 균일하게 펼쳐 선명도를 극대화합니다.
- **핵심 개념**:
  - **누적 분포 함수(CDF) 기반 평활화**: `cv2.equalizeHist()`는 전체 명도 범위를 골고루 활용하도록 변환합니다.
  - **컬러 영상 평활화의 주의점**: B, G, R 채널을 개별 평활화하면 색상 밸런스가 무너집니다. 따라서 **YCrCb** 색 공간으로 변환 후 밝기 정보인 **Y(Luminance)** 채널에만 평활화를 적용하고 다시 BGR로 합성해야 자연스러운 결과가 나옵니다.

---

### 3. 히스토그램 및 색 공간 분할

#### [`ex_histogram.py`](ex_histogram.py) - 흑백 히스토그램 직접 그리기
- **기능**: 이미지의 명암 분포를 계산하여 이미지 우측에 선(Line) 그래프로 렌더링 후 가로 결합(`np.hstack`)하여 표시합니다.
- **핵심 개념**:
  - `cv2.calcHist([img], [0], None, [256], [0, 256])`: 0~255 빈(Bin)별 빈도수를 연산합니다.
  - `cv2.normalize()`: 화면 높이에 맞게 빈도수 값을 0~H 범위로 정규화합니다.

#### [`ex_inrange2.py`](ex_inrange2.py) - HSV 기반 색상 검출
- **기능**: 트랙바를 이용해 원하는 색상 범위(Hue)를 실시간으로 필터링하여 이진 마스크를 생성합니다.
- **핵심 개념**:
  - **RGB vs HSV**: RGB는 조명 변화에 따라 색상과 밝기가 동시에 변해 특정 색을 분리하기 어렵습니다. 반면 HSV는 **H(색상, 0~179)**, **S(채도, 0~255)**, **V(명도, 0~255)**로 분리되어 조명 변화에 훨씬 강인합니다.
  - `cv2.inRange()`: 지정한 하한값과 상한값 사이에 속하는 픽셀만 255(흰색), 나머지는 0(검은색)인 마스크 영상을 만듭니다.

---

### 4. 영상 합성 및 기하학적 변환

#### [`ex_chroma_key.py`](ex_chroma_key.py) - 실시간 동영상 크로마키 합성
- **기능**: 녹색 배경의 인물 동영상에서 배경을 투명하게 제거하고 다른 배경 동영상(비오는 배경)을 실시간으로 합성합니다.
- **핵심 개념**:
  - `cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)`로 변환 후 초록색 영역(`H_min ~ H_max`, `S_min ~ 255`)을 `cv2.inRange()`로 마스킹합니다.
  - `cv2.copyTo(frame2, mask, frame1)`: 마스크 영역에만 배경 영상의 픽셀을 복사하여 자연스럽게 합성합니다.
  - 스페이스바 토글 및 동영상 끝 도달 시 `cap.set(cv2.CAP_PROP_POS_FRAMES, 0)`을 통한 무한 반복 루프 구현.

#### [`ex_docuscan.py`](ex_docuscan.py) - 원근 투시 변환 문서 스캐너
- **기능**: 기울어진 문서나 도로 이미지에서 4개의 모서리 점을 마우스 드래그로 조작하고, Enter 키 입력 시 직사각형 A4 비율 영상으로 똑바르게 펼쳐줍니다.
- **핵심 개념**:
  - `cv2.setMouseCallback()`: 마우스 클릭 및 드래그 이벤트를 캡처하여 ROI 다각형 좌표를 실시간 이동.
  - **투시 변환 (Perspective Transform, Homography)**: 3차원 공간상의 4개 점을 평면의 직사각형 4개 점으로 매핑하는 $3 \times 3$ 변환 행렬을 계산합니다.
    - `cv2.getPerspectiveTransform(srcQuad, dstQuad)`
    - `cv2.warpPerspective(src, pers, (dw, dh))`

---

### 5. 딥러닝 기반 얼굴 및 표정 인식

#### [`ex_face.py`](ex_face.py) - YuNet + MobileFaceNet + 실시간 이모지 오버레이
- **기능**: 웹캠 실시간 영상에서 고속으로 얼굴을 검출하고, 5개 랜드마크(양 눈, 코, 입꼬리)로 정렬한 뒤 7가지 감정을 추론하여 해당 감정의 애플 컬러 이모지와 텍스트를 오버레이합니다.
- **사용 모델 (OpenCV Zoo ONNX)**:
  - `face_detection_yunet_2023mar.onnx`: 경량 고성능 얼굴 감지 모델 (`cv2.FaceDetectorYN`)
  - `facial_expression_recognition_mobilefacenet_2022july.onnx`: 7종 표정 분류 모델 (`cv2.dnn.readNet`)
- **핵심 개념**:
  - 모델 파일이 로컬에 없으면 GitHub OpenCV Zoo 저장소에서 자동 다운로드 수행.
  - **얼굴 정렬 (Face Alignment)**: 눈, 코, 입 위치를 표준 랜드마크 좌표(`STD_POINTS`)로 아핀 변환(`cv2.estimateAffinePartial2D`, `cv2.warpAffine`)하여 포즈 변화에 강인하게 인식.
  - **알파 블렌딩 (Alpha Blending)**: macOS의 `Apple Color Emoji.ttc` 폰트로 생성된 투명 RGBA 이모지를 BGR 화면에 부드럽게 합성.

---

### 6. 생성형 AI (Gemini) Function Calling 연동

#### [`ex_gemini_brightness.py`](ex_gemini_brightness.py) - LLM Tool 기반 지능형 밝기 제어
- **기능**: 최신 `google-genai` SDK를 활용하여, 사용자의 자연어 프롬프트를 `gemini-3.8-flash` 모델이 분석하고 `adjust_brightness` 함수를 도구(Tool)로 자동 호출하여 OpenCV 처리를 수행합니다.
- **핵심 개념**:
  - **Function Calling (Tool Use)**: LLM이 직접 픽셀을 조작하는 대신, 함수 시그니처와 Docstring을 이해하고 정확한 인자(`offset`)를 산출하여 Python 함수를 트리거합니다.
  - 자연어 요청 예시:
    - `"images/2.jpg 이미지의 밝기를 40만큼 밝게 조절해줘"` -> 모델이 `adjust_brightness(image_path="images/2.jpg", offset=40)` 호출.
    - `"이 사진이 너무 어두워 보여. 알맞게 밝혀줘"` -> 모델이 문맥을 분석하여 적절한 오프셋 결정 후 실행.

---

### 7. 유틸리티

#### [`convert_webp_to_jpg.py`](convert_webp_to_jpg.py) - WebP to JPG 변환기
- **기능**: WebP 포맷 이미지를 고품질 JPG 파일로 변환합니다 (단일 파일 및 폴더 일괄 변환 지원).
- **핵심 개념**:
  - Pillow(`PIL.Image`)를 활용하여 알파 채널(RGBA, 투명 배경)이 존재하는 WebP 이미지를 흰색 배경 위에 자연스럽게 합성 후 JPEG으로 저장합니다.

---

## 💡 핵심 컴퓨터 비전 개념 요약

| 개념 | 주요 OpenCV API | 설명 |
| :--- | :--- | :--- |
| **포화 연산 (Saturation)** | `cv2.add()`, `cv2.convertScaleAbs()` | `uint8` 범위(0~255)를 벗어날 때 오버플로우/언더플로우를 방지하고 최소/최댓값으로 클리핑 |
| **색 공간 (Color Spaces)** | `cv2.cvtColor()` | BGR(디스플레이), HSV(색상/채도 분리 및 마스킹), YCrCb(밝기/색차 분리) |
| **히스토그램 평활화** | `cv2.equalizeHist()` | 픽셀 명암 분포를 균일하게 분산시켜 영상의 대비(Contrast)를 개선 |
| **임계값/범위 분할** | `cv2.inRange()` | 특정 조건(색상, 밝기 범위)에 부합하는 영역을 이진 마스크(0과 255)로 추출 |
| **투시 변환 (Perspective)** | `cv2.getPerspectiveTransform()`, `cv2.warpPerspective()` | 원근감이 적용된 4점 좌표를 직사각형 평면으로 재매핑 (Homography) |
| **DNN 모듈 / ONNX** | `cv2.dnn.readNet()`, `cv2.FaceDetectorYN` | 사전 학습된 딥러닝 ONNX 모델을 OpenCV 런타임에서 직접 고속 추론 |
| **AI 에이전트 연동** | `google.genai`, `client.chats.create(tools=[...])` | LLM의 의미론적 판단(Reasoning) 능력과 OpenCV의 결정론적 영상 처리 기능 결합 |

---

## 🚀 실행 예시 모음

```bash
# 1. 트랙바 밝기 조절
python ex_brightness_trackbar.py

# 2. 동영상 크로마키 합성 (Space: 합성 토글, ESC: 종료)
python ex_chroma_key.py

# 3. 문서 스캐너 (모서리 드래그 후 Enter: 변환, ESC: 종료)
python ex_docuscan.py

# 4. 얼굴 감지 및 표정 인식 (웹캠 필요)
python ex_face.py

# 5. Gemini AI 자연어 밝기 조절
python ex_gemini_brightness.py "images/home.jpg 이미지의 밝기를 30만큼 어둡게 만들어줘"

# 6. WebP 이미지 일괄 변환
python convert_webp_to_jpg.py images/
```
