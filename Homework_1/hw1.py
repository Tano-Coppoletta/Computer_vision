import cv2
import numpy as np


def correlation(im, kernel):
    im_height, im_width, im_channels = im.shape
    kernel_size = kernel.shape[0]
    pad_size = int((kernel_size-1)/2)
    im_padded = np.zeros((im_height+pad_size*2, im_width+pad_size*2, im_channels))
    im_padded[pad_size:-pad_size, pad_size:-pad_size, :] = im

    im_out = np.zeros_like(im)
    for c in range(im_channels):
        for x in range(im_width):
            for y in range(im_height):
                im_patch = im_padded[y:y+kernel_size, x:x+kernel_size, c]
                new_value = np.sum(kernel*im_patch)
                new_value = (int)(min(max(0, new_value), 255))  # bound the pixel within (0, 255)
                im_out[y, x, c]= new_value
    return im_out

#this function flip the kernel and calls correlation that actually does the job
def convolution(im,kernel):
    return correlation(im,np.flip(kernel))


#mean filter
def mean_filter(im,kernel_size):
    kernel = np.ones((kernel_size,kernel_size), dtype=float)/kernel_size**2
    #print(kernel)

    im_out = convolution(im, kernel)

    return im_out

def sharpening_filter(im, kernel_size):
    alpha=2

    center=(int(kernel_size/2))
    matrix1 = np.zeros((kernel_size, kernel_size), dtype=float)
    matrix1[center][center]=alpha

   # print(matrix1)
    matrix2 = np.ones((kernel_size,kernel_size), dtype=float)/kernel_size**2

    #print(matrix2)

    kernel = matrix1-matrix2

    im_out = convolution(im, kernel)
    return im_out

def gaussian_filter(im, kernel_size):
    std=1
    x1 = 1/(2*np.pi*std**2)
    k=int((kernel_size-1)/2)
    gaussian_kern = np.zeros((kernel_size,kernel_size),dtype=float)
    for x in range(-k,k+1):
        for y in range(-k,k+1):
            x2=np.exp(-(x**2+y**2)/(2*std**2))
            gaussian_kern[x+k][y+k]=x1*x2

    im_out = convolution(im,gaussian_kern)
    return im_out

#P2
def median_filter(im,kernel_size):
    im_height, im_width, im_channels = im.shape
    im_out = np.zeros_like(im)

    k=int((kernel_size-1)/2)

    for c in range(im_channels):
        for j in range(im_width):
            for i in range(im_height):
                start_i = i-k
                start_j=j-k
                end_i=i+k
                end_j=j+k
                if(i-k<=0):
                    start_i=0
                if(j - k <= 0):
                    start_j=0

                if(i+k>=im_height):
                    end_i=im_height

                if(j+k>=im_width):
                    end_j=im_width

                new_value = np.median(im[start_i:end_i,start_j:end_j])


                new_value = (int)(min(max(0, new_value), 255))  # bound the pixel within (0, 255)
                im_out[i, j, c] = new_value
    return im_out

#P3
def use_filter2D(im, kernel_size):
    std=1

    x1 = 1 / (2 * np.pi * std ** 2)
    k = int((kernel_size - 1) / 2)
    gaussian_kern = np.zeros((kernel_size, kernel_size), dtype=float)
    for x in range(-k, k + 1):
        for y in range(-k, k + 1):
            x2 = np.exp(-(x ** 2 + y ** 2) / (2 * std ** 2))
            gaussian_kern[x + k][y + k] = x1 * x2
    im_out = cv2.filter2D(im, -1,gaussian_kern)
    return im_out


im = cv2.imread("lena.png")
im = im.astype(float)

#P1
for kernel_size in (3,5,7):
    im_out = mean_filter(im,kernel_size)
    im_out = im_out.astype(np.uint8)
    cv2.imwrite('hw1_mean_filter'+str(kernel_size)+'.png', im_out)

    im_out = sharpening_filter(im,kernel_size)
    im_out= im_out.astype(np.uint8)
    cv2.imwrite('hw1_sharpening_filter'+str(kernel_size)+'.png', im_out)

    im_out = gaussian_filter(im,kernel_size)
    im_out = im_out.astype(np.uint8)
    cv2.imwrite('hw1_gaussian_filter'+str(kernel_size)+'.png', im_out)

#P2
art = cv2.imread("art.png")
art = art.astype(float)
for kernel_size in (3,5,7,9):
    im_out = median_filter(art,kernel_size)
    im_out = im_out.astype(np.uint8)
    cv2.imwrite('hw1_median_filter_art'+str(kernel_size)+'.png', im_out)

    im_out = mean_filter(art,kernel_size)
    im_out = im_out.astype(np.uint8)
    cv2.imwrite('hw1_mean_filter_art'+str(kernel_size)+'.png', im_out)

#P3
for kernel_size in (3,5,7):
    im_out= use_filter2D(im, kernel_size)
    im_out = im_out.astype(np.uint8)
    cv2.imwrite('hw1_filter2D_'+str(kernel_size)+'.png', im_out)



