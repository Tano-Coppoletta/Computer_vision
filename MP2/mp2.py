import cv2
import numpy as np
import matplotlib.pyplot as plt
import math

def hysteresis_thresholding(im, low_threshold, high_threshold):
    M,N = im.shape
    im_out = np.zeros((M,N))
    #strong edges
    strong_edge_i, strong_edge_j = np.where(im>high_threshold)
    #noise
    noise_i,noise_j = np.where(im<low_threshold)
    #weak edges
    weak_i,weak_j = np.where((im<=high_threshold) & (im>=low_threshold))
    im_out[strong_edge_i,strong_edge_j] = 255
    im_out[noise_i,noise_j] = 125
    im_out[weak_i,weak_j] = 0
    return im_out

def edge_linking(im):
    M,N = im.shape
    im_out = np.zeros((M,N))
    for i in range(1,M-1):
        for j in range(1,N-1):
            if(im[i,j]==125):
                if 255 in [im[i-1,j-1],im[i-1,j],im[i-1,j+1],im[i,j-1],im[i,j+1],im[i+1,j-1],im[i+1,j],im[i+1,j+1]]:
                    im_out[i,j]=255
    return im_out


def convolution(im, kernel):
    kernel = np.flip(np.flip(kernel, axis=0), axis=1)
    im_height, im_width = im.shape
    kernel_size = kernel.shape[0]
    pad_size = int((kernel_size - 1) / 2)
    im_padded = np.zeros((im_height + pad_size * 2, im_width + pad_size * 2))
    im_padded[pad_size:-pad_size, pad_size:-pad_size] = im

    im_out = np.zeros_like(im)
    for x in range(im_width):
        for y in range(im_height):
            im_patch = im_padded[y:y + kernel_size, x:x + kernel_size]
            new_value = np.sum(kernel * im_patch)
            im_out[y, x] = new_value
    return im_out


def get_gaussian_kernel(kernel_size, sigma):
    kernel_x = np.linspace(-(kernel_size // 2), kernel_size // 2, kernel_size)
    for i in range(kernel_size):
        kernel_x[i] = np.exp(-(kernel_x[i] / sigma) ** 2 / 2)
    kernel = np.outer(kernel_x.T, kernel_x.T)

    kernel *= 1.0 / kernel.sum()
    return kernel


def compute_gradient(im):
    sobel_filter_x = np.array([[1, 0, -1], [2, 0, -2], [1, 0, -1]])
    sobel_filter_y = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])
    gradient_x = convolution(im, sobel_filter_x)
    gradient_y = convolution(im, sobel_filter_y)

    magnitude = np.sqrt(np.square(gradient_x) + np.square(gradient_y))
    magnitude *= 255.0 / magnitude.max()
    direction = np.arctan2(gradient_y, gradient_x)
    direction *= 180 / np.pi
    return magnitude, direction


def nms(magnitude, direction):
    height, width = magnitude.shape
    res = np.zeros(magnitude.shape)
    direction[direction < 0] += 180  # (-180, 180) -> (0, 180)

    for y in range(1, height - 1):
        for x in range(1, width - 1):
            current_direction = direction[y, x]
            current_magnitude = magnitude[y, x]
            if (0 <= current_direction < 22.5) or (157.5 <= current_direction <= 180):
                p = magnitude[y, x - 1]
                r = magnitude[y, x + 1]

            elif 22.5 <= current_direction < 67.5:
                p = magnitude[y + 1, x + 1]
                r = magnitude[y - 1, x - 1]

            elif 67.5 <= current_direction < 112.5:
                p = magnitude[y - 1, x]
                r = magnitude[y + 1, x]

            else:
                p = magnitude[y - 1, x + 1]
                r = magnitude[y + 1, x - 1]

            if current_magnitude >= p and current_magnitude >= r:
                res[y, x] = current_magnitude

    return res

def HoughTransform(edge_map):
    theta_values = np.deg2rad(np.arange(-90.0, 90.0))
    height, width = edge_map.shape
    diagonal_length = int(round(math.sqrt(width * width + height * height)))
    rho_values = np.linspace(-diagonal_length, diagonal_length, diagonal_length * 2 + 1)

    accumulator = np.zeros((len(rho_values), len(theta_values)), dtype=int)
    y_coordinates, x_coordinates = np.nonzero(edge_map)

    for edge_idx in range(len(x_coordinates)):
        x = x_coordinates[edge_idx]
        y = y_coordinates[edge_idx]
        for theta_idx in range(len(theta_values)):
            theta = theta_values[theta_idx]
            rho = int(round(x * np.cos(theta) + y * np.sin(theta)))
            accumulator[rho + diagonal_length, theta_idx] += 1
        # print("%d out of %d edges have voted" % (edge_idx+1, len(x_coordinates)))
        # cv2.imshow("Accumulator", (accumulator*255/accumulator.max()).astype(np.uint8))
        # cv2.waitKey(0)
    return accumulator, theta_values, rho_values



im = cv2.imread("lena.png", 0)
im = im.astype(float)

gaussian_kernel = get_gaussian_kernel(9, 3)
im_smoothed = convolution(im, gaussian_kernel)

#cv2.imshow("Original image", im.astype(np.uint8))
#cv2.imshow("Smoothed image", im_smoothed.astype(np.uint8))
#cv2.waitKey()
#cv2.destroyAllWindows()

gradient_magnitude, gradient_direction = compute_gradient(im_smoothed)

edge_nms = nms(gradient_magnitude, gradient_direction)

after_hysteresis=hysteresis_thresholding(edge_nms,0.05*255,0.1*255)
after_hysteresis = after_hysteresis.astype(np.uint8)
cv2.imwrite('hysteresis_1' + '.png', after_hysteresis)

after_edge_linking = edge_linking(after_hysteresis)
after_edge_linking = after_edge_linking.astype(np.uint8)
cv2.imwrite('after_edge_linking_1'+ '.png', after_edge_linking)


im = cv2.imread('shape.bmp')

im_gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)

#CHANGE
edge_map = cv2.Canny(im_gray, 70, 150)

accumulator, theta_values, rho_values = HoughTransform(edge_map)


lines = np.argwhere(accumulator > 30)



height, width = im_gray.shape
for line in lines:
    rho = rho_values[line[0]]
    theta = theta_values[line[1]]
    slope = -np.cos(theta)/np.sin(theta)
    intercept = rho/np.sin(theta)
    x1, x2 = 0, width
    y1 = int(slope*x1 + intercept)
    y2 = int(slope*x2 + intercept)
    cv2.line(im, (x1, y1), (x2, y2), (0, 0, 255), 2)

cv2.imshow("Edges", edge_map)
cv2.imshow("Hough Transform", (accumulator*255/accumulator.max()).astype(np.uint8))
cv2.imshow("Output", im)
cv2.waitKey(0)
cv2.destroyAllWindows()

