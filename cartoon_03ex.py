import cv2
import numpy as np

# 이미지 로드
img = cv2.imread('cartoon_rendering_original.jpg')
if img is None:
    print("이미지를 찾을 수 없습니다.")
    exit()

# 1️⃣ **엣지 검출 (Canny)**
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # 흑백 변환
edges = cv2.Canny(gray, 100, 200)  # 엣지 검출
edges = cv2.GaussianBlur(edges, (5, 5), 0)  # 엣지를 부드럽게 만듦
edges_inv = cv2.bitwise_not(edges)  # 엣지를 반전

# 2️⃣ **색상 단순화 (Bilateral 필터)**
color = cv2.bilateralFilter(img, 9, 75, 75)
color = cv2.bilateralFilter(color, 9, 75, 75)

# 3️⃣ **샤프닝 (Sharpening)**
sharpening_kernel = np.array([[0, -1, 0],
                               [-1, 5, -1],
                               [0, -1, 0]])  # 샤프닝 커널
sharpened = cv2.filter2D(color, -1, sharpening_kernel)

# 4️⃣ **엣지와 색상 결합**
cartoon = cv2.bitwise_and(sharpened, sharpened, mask=edges_inv)

# 결과 출력
cv2.imshow("Cartoon", cartoon)
cv2.waitKey(0)
cv2.destroyAllWindows()
