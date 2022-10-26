import os

import numpy as np
import cv2
import math


def calculate_hs_histogram(img, hs_hist, bin_size):
    height, width, _ = img.shape
    img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    for i in range(height):
        for j in range(width):
            h = img_hsv[i, j, 0]
            s = img_hsv[i, j, 1]
            hs_hist[math.floor(h / bin_size), math.floor(s / bin_size)] += 1
    return hs_hist


def color_segmentation(img, hs_hist, bin_size, threshold):
    height, width, _ = img.shape
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    mask = np.zeros((height, width, 1))
    for i in range(height):
        for j in range(width):
            h = hsv[i, j, 0]
            s = hsv[i, j, 1]
            if hs_hist[math.floor(h / bin_size), math.floor(s / bin_size)] > threshold:
                mask[i, j, 0] = 1
    return mask


def calculate_mean_and_covariance():
    h_vector = []
    s_vector = []
    for i in range(1, 11):
        img_train = cv2.imread(f"./skin_patches/skin{i}.jpg")
        img_hsv = cv2.cvtColor(img_train, cv2.COLOR_BGR2HSV)
        h, s = img_hsv[:, :, 0], img_hsv[:, :, 1]
        h_vector = np.concatenate((np.asarray(h_vector), np.asarray(h).reshape(-1)))
        s_vector = np.concatenate((np.asarray(s_vector), np.asarray(s).reshape(-1)))

    m = np.matrix((h_vector, s_vector))
    mean_vector = np.mean(m, axis=1, dtype=np.float64)
    # print(mean_vector.shape)
    covariance_matrix = np.cov(m)
    #print(covariance_matrix)
    return mean_vector, covariance_matrix


def compute_p(img, mean, covariance, threshold):
    img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, s = img_hsv[:, :, 0], img_hsv[:, :, 1]
    height, width, _ = img.shape
    mask = np.zeros((height, width, 1))
    for i in range(height):
        for j in range(width):
            f = 1 / (np.sqrt((2 * np.pi) ** 2 * np.linalg.det(covariance)))
            e = np.exp(-0.5 * np.dot(np.dot((img_hsv[i, j, :2] - mean.T), np.linalg.inv(covariance)) ,
                    np.array([[img_hsv[i, j, 0]], [img_hsv[i,j,1]]]) - mean))
            #print( np.array([img_hsv[i, j, 0], img_hsv[i,j,1]]))
            p = f * e
            #print(p)
            if p > threshold:
                mask[i, j, 0] = 1

    return mask


# P1
bin_size = 20
max_h = 179
max_s = 255
hs_hist = np.zeros((math.ceil((max_h + 1) / bin_size), math.ceil((max_s + 1) / bin_size)))
#
# # Training
for i in range(1, 11):
    img_train = cv2.imread(f"./skin_patches/skin{i}.jpg")
    hs_hist = calculate_hs_histogram(img_train, hs_hist, bin_size)
hs_hist /= hs_hist.sum()

# Testing
img_test = cv2.imread("testing_image.bmp")

threshold = 0.03
mask = color_segmentation(img_test, hs_hist, bin_size, threshold)

img_seg = img_test * mask

cv2.imwrite(f"./outputs/input1.png", img_test)
cv2.imwrite("./outputs/mask1.png", (mask * 255).astype(np.uint8))
cv2.imwrite("./outputs/segmentation1.png", img_seg.astype(np.uint8))


# P2
threshold = 0.000002

mean, covariance = calculate_mean_and_covariance()
img_test = cv2.imread("testing_image.bmp")
mask = compute_p(img_test, mean, covariance, threshold)
img_seg = img_test * mask
cv2.imwrite("./outputs/mask2.png", (mask*255).astype(np.uint8))
cv2.imwrite("./outputs/segmentation2.png", img_seg.astype(np.uint8))

#P3
img = cv2.imread("checkerboard.png")
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
gray = np.float32(gray)
dst = cv2.cornerHarris(gray,2,3,0.04)
#result is dilated for marking the corners, not important
dst = cv2.dilate(dst,None)

# Threshold for an optimal value, it may vary depending on the image.
img[dst>0.01*dst.max()]=[0,0,255]
cv2.imwrite("./outputs/p3_1.png",img)


img = cv2.imread("toy.png")
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

gray = np.float32(gray)
dst = cv2.cornerHarris(gray,4,3,0.04)
#result is dilated for marking the corners, not important
dst = cv2.dilate(dst,None)

# Threshold for an optimal value, it may vary depending on the image.
img[dst>0.01*dst.max()]=[0,0,255]
cv2.imwrite("./outputs/p3_3.png",img)