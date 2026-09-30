import os
import sys
import cv2
from google import genai
from google.genai import types


def adjust_brightness(
    image_path: str, offset: int, output_path: str = "images/output_brightness.jpg"
) -> str:
    """OpenCV convertScaleAbs를 사용하여 이미지의 밝기를 조절합니다.

    Args:
        image_path: 원본 이미지 파일 경로 (예: 'images/2.jpg')
        offset: 밝기 조절 값 (-100 ~ 100). 양수는 밝게, 음수는 어둡게 조절합니다.
        output_path: 결과 이미지를 저장할 경로 (기본값: 'images/output_brightness.jpg')

    Returns:
        작업 성공 여부 및 결과 메시지
    """
    src = cv2.imread(image_path)
    if src is None:
        return f"Error: '{image_path}' 이미지를 불러올 수 없습니다."

    # ex_brightness_trackbar.py와 동일한 방식의 밝기 조절
    dst = cv2.convertScaleAbs(src, alpha=1.0, beta=offset)
    cv2.imwrite(output_path, dst)
    return f"Success: 밝기를 {offset}만큼 조절하여 '{output_path}'에 저장했습니다."


def main():
    if not os.environ.get("GEMINI_API_KEY"):
        print("Error: GEMINI_API_KEY 환경변수가 설정되어 있지 않습니다.")
        sys.exit(1)

    client = genai.Client()

    # gemini-3.8-flash 모델과 adjust_brightness 툴 등록
    chat = client.chats.create(
        model="gemini-3.8-flash",
        config=types.GenerateContentConfig(
            tools=[adjust_brightness],
            temperature=0.0,
        ),
    )

    # 명령행 인자가 있으면 해당 요청을 사용하고, 없으면 기본 요청 실행
    prompt = (
        sys.argv[1]
        if len(sys.argv) > 1
        else "images/2.jpg 이미지의 밝기를 40만큼 밝게 조절해줘."
    )

    print(f"[User]: {prompt}\n")
    response = chat.send_message(prompt)
    print(f"[Gemini]: {response.text}\n")

    # 결과 이미지가 생성되었으면 창으로 확인
    output_path = "images/output_brightness.jpg"
    if os.path.exists(output_path):
        result_img = cv2.imread(output_path)
        if result_img is not None:
            cv2.namedWindow("Gemini Brightness Result", cv2.WINDOW_NORMAL)
            cv2.imshow("Gemini Brightness Result", result_img)
            print("결과 이미지를 표시합니다. 아무 키나 누르면 종료됩니다.")
            cv2.waitKey(0)
            cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
