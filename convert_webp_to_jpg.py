import os
import sys
from pathlib import Path
from PIL import Image


def convert_webp_to_jpg(input_path: str | Path, output_path: str | Path = None, quality: int = 95) -> Path:
    """
    WebP 이미지를 JPG 이미지로 변환합니다.
    - 투명 배경(RGBA)이 있는 경우 흰색 배경으로 자연스럽게 합성합니다.
    - quality: 저장 품질 (1~100, 기본값 95)
    """
    input_path = Path(input_path)
    if not input_path.exists():
        raise FileNotFoundError(f"파일을 찾을 수 없습니다: {input_path}")

    if output_path is None:
        output_path = input_path.with_suffix(".jpg")
    else:
        output_path = Path(output_path)

    with Image.open(input_path) as img:
        # 투명도(알파 채널)가 있는 경우 흰색 배경 위에 합성
        if img.mode in ("RGBA", "LA") or (img.mode == "P" and "transparency" in img.info):
            rgba_img = img.convert("RGBA")
            background = Image.new("RGB", rgba_img.size, (255, 255, 255))
            background.paste(rgba_img, mask=rgba_img.split()[3])
            background.save(output_path, "JPEG", quality=quality)
        else:
            img.convert("RGB").save(output_path, "JPEG", quality=quality)

    print(f"변환 완료: {input_path.name} -> {output_path.name}")
    return output_path


def convert_directory(dir_path: str | Path = ".", quality: int = 95):
    """지정된 폴더 내의 모든 .webp 파일을 .jpg로 일괄 변환합니다."""
    dir_path = Path(dir_path)
    webp_files = list(dir_path.glob("*.webp"))

    if not webp_files:
        print(f"변환할 .webp 파일이 없습니다: {dir_path.resolve()}")
        return

    print(f"총 {len(webp_files)}개의 .webp 파일을 변환합니다...")
    for webp_file in webp_files:
        convert_webp_to_jpg(webp_file, quality=quality)


if __name__ == "__main__":
    # 실행 인자가 있으면 해당 파일/폴더 변환, 없으면 현재 폴더 내 webp 파일 일괄 변환
    if len(sys.argv) > 1:
        target = Path(sys.argv[1])
        if target.is_dir():
            convert_directory(target)
        else:
            convert_webp_to_jpg(target)
    else:
        # 기본 동작: 현재 폴더의 webp 파일 일괄 변환
        convert_directory(".")
