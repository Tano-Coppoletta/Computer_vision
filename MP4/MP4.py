import numpy as np
import cv2
import math

#create three channel white image
white_img = np.zeros([500,500,3],dtype=np.uint8)
white_img.fill(255)

a = (100,40)
radius = 10
red = (0,0,255) #red
thickness = -1

image= cv2.circle(white_img, a, radius, red, thickness)
cv2.imwrite("image1.png",image)

matrix = [[math.cos(math.pi/3), -math.sin(math.pi/3)],
          [math.sin(math.pi/3),math.cos(math.pi/3)]]

b = np.dot(matrix,a)

green = (0,255,0)

image2 = cv2.circle(image, np.round(b).astype(int) , radius, green, thickness)
cv2.imwrite("image2.png", image2)


c = (100,100)
black = (0,0,0)
image3 = cv2.circle(image2, c, radius, black, thickness)
cv2.imwrite("image3.png", image3)

ox, oy = c
px, py = a

qx = ox + math.cos(math.pi/3) * (px - ox) - math.sin(math.pi/3) * (py - oy)
qy = oy + math.sin(math.pi/3) * (px - ox) + math.cos(math.pi/3) * (py - oy)

d = (qx,qy)
blue = (255,0,0)
image4 = cv2.circle(image3, np.round(d).astype(int), radius, blue, thickness)
cv2.imwrite("image4.png",image4)

#P2
#1
img = cv2.imread("lena.png")
rows, cols, ch = img.shape
matrix = np.float32([[1,0,100],[0,1,200]])
image5 = cv2.warpAffine(img, matrix, (cols,rows))
cv2.imwrite("lena1.png",image5)
#2
traslation1 = np.float32([[1,0,-cols/2],[0,1,-rows/2],[0,0,1]])
flip = np.float32([[-1,0,0],[0,1,0],[0,0,1]])
traslation2 = np.float32([[1,0,cols/2],[0,1,rows/2],[0,0,1]])

result_matrix = np.dot(traslation2,flip)
result_matrix = np.dot(result_matrix,traslation1)
image6 = cv2.warpAffine(img, result_matrix[:-1,:], (cols,rows))
cv2.imwrite("lena2.png",image6)

#3
matrix = np.float32([[math.cos(math.pi/4),-math.sin(math.pi/4),0],[math.sin(math.pi/4),math.cos(math.pi/4),0]])
image7 = cv2.warpAffine(img, matrix, (cols, rows))
cv2.imwrite("lena3.png",image7)

#4
#traslate the center of the image to the origin
traslation1 = np.float32([[1, 0, -cols/2],[0, 1, -rows/2],[0,0,1]])
#rotate the image
rotation = np.float32([[math.cos(math.pi/4),-math.sin(math.pi/4),0],[math.sin(math.pi/4),math.cos(math.pi/4),0],[0,0,1]])
#traslate back
traslation2 = np.float32([[1,0,cols/2],[0,1,rows/2],[0,0,1]])
result_matrix = np.dot(traslation2,rotation)
result_matrix = np.dot(result_matrix,traslation1)
image8 = cv2.warpAffine(img, result_matrix[:-1,:], (cols,rows))
cv2.imwrite("lena4.png",image8)



