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
                im_out[y, x, c]= new_value
    return im_out

kernel = np.array([[1,1,1],
                   [1,1,1],
                   [1,1,1]])/9
im = cv2.imread("lena.png")
im = im.astype(float)
im_out = correlation(im, kernel)
im_out = im_out.astype(np.uint8)
cv2.imwrite('output_image.png', im_out)
cv2.imshow("Output", im_out)
cv2.waitKey(0)
cv2.destroyAllWindows()
