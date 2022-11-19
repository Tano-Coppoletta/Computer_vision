
import numpy as np
import cv2
import math

img = cv2.imread("lena.png")
print(img[:10,:10,0])
img = img+1

print(img[:10,:10,0])

gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
gray = np.float32(gray)
dst = cv2.cornerHarris(gray,2,3,0.03)
#result is dilated for marking the corners, not important
dst = cv2.dilate(dst,None)

# Threshold for an optimal value, it may vary depending on the image.
img[dst>0.01*dst.max()]=[0,0,255]
print(img[:10,:10,0])
cv2.imwrite("./lena_scaled.png",img)

