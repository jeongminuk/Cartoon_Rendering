import cv2
import numpy as np

def cartoonize_image(img):
    # 1️⃣ **엣지 검출 (Edge Detection)**
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # 흑백 변환
    gray = cv2.medianBlur(gray, 5)  # 노이즈 제거
    edges = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                  cv2.THRESH_BINARY, 9, 9)  # 윤곽선 강조

    # 2️⃣ **색상 단순화 (Color Quantization)**
    data = np.float32(img).reshape((-1, 3))  # 데이터를 (픽셀 개수, 3) 형태로 변환
    k = 9  # 클러스터 개수 (색상 수)
    _, labels, centers = cv2.kmeans(data, k, None,
                                    (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2),
                                    10, cv2.KMEANS_RANDOM_CENTERS)
    centers = np.uint8(centers)  # 정수형 변환
    quantized = centers[labels.flatten()].reshape(img.shape)  # 색상 재구성

    # 3️⃣ **엣지와 색상 결합**
    cartoon = cv2.bitwise_and(quantized, quantized, mask=edges)

    return cartoon

# 이미지 로드
img = cv2.imread('cartoon_rendering_original.jpg')
cartoon = cartoonize_image(img)

# 결과 출력
cv2.imshow("Cartoon", cartoon)
cv2.imwrite('cartoon_rendering_result.jpg', cartoon)
cv2.waitKey(0)
cv2.destroyAllWindows()