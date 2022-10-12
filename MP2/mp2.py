import cv2
import numpy as np
import matplotlib.pyplot as plt
import math


def hysteresis_thresholding(im, low_threshold, high_threshold):
    M, N = im.shape
    map_strong = np.zeros((M, N))
    map_weak = np.zeros((M, N))
    map_noise = np.zeros((M, N))

    # strong edges
    strong_edge_i, strong_edge_j = np.where(im > high_threshold)
    # noise
    noise_i, noise_j = np.where(im < low_threshold)
    # weak edges
    weak_i, weak_j = np.where((im <= high_threshold) & (im >= low_threshold))
    map_strong[strong_edge_i, strong_edge_j] = 1
    map_noise[noise_i, noise_j] = 1
    map_weak[weak_i, weak_j] = 1
    return map_strong,map_weak,map_noise


def edge_linking(map_strong, map_weak):
    M, N = map_strong.shape
    im_out = map_strong.copy()
    for i in range(1, M - 1):
        for j in range(1, N - 1):
            if (map_weak[i, j] ==1):
                if map_strong[i - 1, j - 1]==1 or im[i - 1, j]==1 or im[i - 1, j + 1]==1 or im[i, j - 1]==1 or im[i, j + 1]==1 or  im[i + 1, j - 1]==1 or im[i + 1, j]==1 or im[i + 1, j + 1]==1:
                    im_out[i, j] = 1
                else:
                    im_out[i, j] = 0
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


def suppress_lines(accumulator, lines, tolerance):
    max = (-1, -1)
    maxs = []

    for i in range(0, len(lines) - 1):
        if i == len(lines) - 2 or (np.all(max != (-1, -1)) and not (
                np.isclose(lines[i, 0], lines[i + 1, 0], atol=tolerance)
                and np.isclose(lines[i, 1], lines[i + 1, 1], atol=tolerance))):
            maxs.append(max)
            max = (-1, -1)
        else:
            if np.all(max == (-1, -1)) or accumulator[lines[i, 0], lines[i, 1]] > accumulator[max[0], max[1]]:
                max = lines[i]
    return maxs


im = cv2.imread("lena.png",0)
im = im.astype(float)

gaussian_kernel = get_gaussian_kernel(9, 3)
im_smoothed = convolution(im, gaussian_kernel)

# cv2.imshow("Original image", im.astype(np.uint8))
# cv2.imshow("Smoothed image", im_smoothed.astype(np.uint8))
# cv2.waitKey()
# cv2.destroyAllWindows()

gradient_magnitude, gradient_direction = compute_gradient(im_smoothed)

edge_nms = nms(gradient_magnitude, gradient_direction)


for low_t,high_t in [(0.05,0.1),(0.09,0.2),(0.0001,0.001)]:

    map_strong, map_weak, map_noise = hysteresis_thresholding(edge_nms, low_t * 255, high_t * 255)
    map_strong = map_strong.astype(np.uint8)*255
    cv2.imwrite(f"map_strong_{low_t}_{high_t}.png", map_strong)

    map_weak = map_weak.astype(np.uint8)*255
    cv2.imwrite(f"map_weak_{low_t}_{high_t}.png", map_weak)

    map_noise = map_noise.astype(np.uint8)*255
    cv2.imwrite(f"map_noise{low_t}_{high_t}.png", map_noise)

    after_edge_linking = edge_linking(map_strong,map_weak)
    after_edge_linking = after_edge_linking.astype(np.uint8)
    cv2.imwrite(f"after_edge_linking_{low_t}_{high_t}.png", after_edge_linking)

im1 = cv2.imread('shape.bmp')
im2 = cv2.imread('paper.bmp')
i=0
for im in [im1,im2]:

    im_gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)

    # CHANGE
    edge_map = cv2.Canny(im_gray, 70, 150)

    accumulator, theta_values, rho_values = HoughTransform(edge_map)

    lines = np.argwhere(accumulator > 40)

    lines = suppress_lines(accumulator, lines,10)

    height, width = im_gray.shape
    for line in lines:
        rho = rho_values[line[0]]
        theta = theta_values[line[1]]
        slope = -np.cos(theta) / np.sin(theta)
        intercept = rho / np.sin(theta)
        x1, x2 = 0, width
        y1 = int(slope * x1 + intercept)
        y2 = int(slope * x2 + intercept)
        cv2.line(im, (x1, y1), (x2, y2), (0, 0, 255), 2)


    if i==0:
        cv2.imwrite('shape_edges' + '.png', edge_map)
        cv2.imwrite('shape_hough_transform' + '.png',  (accumulator * 255 / accumulator.max()).astype(np.uint8))
        cv2.imwrite('shape_output' + '.png', im)
    else:
        cv2.imwrite('paper_edges' + '.png', edge_map)
        cv2.imwrite('paper_hough_transform' + '.png',  (accumulator * 255 / accumulator.max()).astype(np.uint8))
        cv2.imwrite('paper_output' + '.png', im)
    i+=1


#using canny

im = cv2.imread("lena.png")
im_gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
out_canny = cv2.Canny(im_gray,0.05*255,0.1*255)
cv2.imwrite('point_4' + '.png', out_canny)

im = cv2.imread("shape.bmp")
im_gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
edges = cv2.Canny(im_gray, 50,180)
lines = cv2.HoughLines(edges, 1, np.pi/180, 30)

for r_theta in lines:
    arr = np.array(r_theta[0], dtype=np.float64)
    r, theta = arr
    a = np.cos(theta)
    b = np.sin(theta)
    x0 = a * r
    y0 = b * r
    x1 = int(x0 + 1000 * (-b))
    y1 = int(y0 + 1000 * (a))
    x2 = int(x0 - 1000 * (-b))
    y2 = int(y0 - 1000 * (a))
    cv2.line(im, (x1, y1), (x2, y2), (0, 0, 255), 2)

cv2.imwrite('point_4_shape.png',im)


im = cv2.imread("paper.bmp")
im_gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
edges = cv2.Canny(im_gray, 50,180)
lines = cv2.HoughLines(edges, 1, np.pi/180, 30)

for r_theta in lines:
    arr = np.array(r_theta[0], dtype=np.float64)
    r, theta = arr
    a = np.cos(theta)
    b = np.sin(theta)
    x0 = a * r
    y0 = b * r
    x1 = int(x0 + 1000 * (-b))
    y1 = int(y0 + 1000 * (a))
    x2 = int(x0 - 1000 * (-b))
    y2 = int(y0 - 1000 * (a))
    cv2.line(im, (x1, y1), (x2, y2), (0, 0, 255), 2)
cv2.imwrite('point_4_paper.png',im)
