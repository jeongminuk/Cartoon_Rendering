import cv2
import numpy as np
import argparse

def cartoonize_image(input_path, output_path):
    # 이미지 로드
    img = cv2.imread(input_path)
    if img is None:
        print(f"이미지를 찾을 수 없습니다: {input_path}")
        return

    # 1️⃣ **엣지 검출**
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # 흑백 변환
    gray = cv2.medianBlur(gray, 5)  # 노이즈 제거
    edges = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                  cv2.THRESH_BINARY, 9, 9)  # 윤곽선 강조

    # 2️⃣ **색상 단순화**
    data = np.float32(img).reshape((-1, 3))  # 픽셀 데이터 변환
    k = 9  # 색상 수
    _, labels, centers = cv2.kmeans(data, k, None,
                                    (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2),
                                    10, cv2.KMEANS_RANDOM_CENTERS)
    centers = np.uint8(centers)
    quantized = centers[labels.flatten()].reshape(img.shape)  # 색상 재구성

    # 3️⃣ **엣지와 색상 결합**
    cartoon = cv2.bitwise_and(quantized, quantized, mask=edges)

    # 결과 저장
    cv2.imwrite(output_path, cartoon)
    print(f"만화 스타일 이미지가 저장되었습니다: {output_path}")

# 명령줄에서 실행 가능하도록 설정
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="이미지를 만화 스타일로 변환하는 프로그램")
    parser.add_argument("input", help="입력 이미지 파일 경로")
    parser.add_argument("output", help="출력 이미지 파일 경로")
    args = parser.parse_args()

    cartoonize_image(args.input, args.output)