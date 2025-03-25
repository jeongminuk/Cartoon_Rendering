import cv2
import numpy as np

# 이미지 로드
img = cv2.imread('gear2.jpeg')
if img is None:
    print("이미지를 찾을 수 없습니다.")
    exit()

# 1️⃣ **엣지 검출 (Laplacian & Canny)**
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # 흑백 변환

# Laplacian 필터 적용 (2차 미분을 통한 윤곽 강조)
laplacian = cv2.Laplacian(gray, cv2.CV_64F, ksize=5)
laplacian = np.uint8(np.abs(laplacian))  # 절댓값 변환 (양수 범위)

# Canny 엣지 검출 추가 (윤곽선을 더 선명하게)
edges = cv2.Canny(laplacian, 50, 150)  # 최솟값 50, 최댓값 150

# 2️⃣ **색상 단순화 (Bilateral 필터)**
color = cv2.bilateralFilter(img, 9, 75, 75)  # 부드러운 색감 유지 (1차 적용)
color = cv2.bilateralFilter(color, 9, 75, 75)  # 한 번 더 적용하여 더 자연스럽게

# 3️⃣ **샤프닝 (Sharpening)**
sharpening_kernel = np.array([[0, -1, 0],
                               [-1, 5, -1],
                               [0, -1, 0]])  # 샤프닝 커널
sharpened = cv2.filter2D(color, -1, sharpening_kernel)  # 샤프닝 적용

# 4️⃣ **엣지와 색상 결합**
cartoon = cv2.bitwise_and(sharpened, sharpened, mask=edges)

# 결과 출력
cv2.imshow("Cartoon", cartoon)
cv2.waitKey(0)
cv2.destroyAllWindows()
