import cv2
import numpy as np

img = cv2.imread('gear2.jpeg')

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
gray = cv2.medianBlur(gray, 5)
edges = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                              cv2.THRESH_BINARY, 9, 9)
color = cv2.bilateralFilter(img, 9, 300, 300)
cartoon = cv2.bitwise_and(color, color, mask=edges)

cv2.imshow("Cartoon", cartoon)
cv2.imwrite('gear2_cartoon.jpg', cartoon)
cv2.waitKey(0)
cv2.destroyAllWindows()