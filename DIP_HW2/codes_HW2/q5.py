import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from PIL import Image
import cv2
import math

###### picture 1 ###########
img = cv2.imread('ghost_2.png')
data=np.asarray(img)
imgplot = plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
print(data)
print(data.shape)
plt.title("original image")
# plt.figure()
plt.show()

# img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# print(img_gray)
# print(img_gray.shape)
# plt.imshow(img_gray)
# plt.title("original image  gray_scale")
# plt.show()
# gray = np.asarray(img_gray)

histogram_array = np.bincount(img.flatten(), minlength=256)

        # normalize
num_pixels = np.sum(histogram_array)
histogram_array = histogram_array / num_pixels

# normalized cumulative histogram
chistogram_array = np.cumsum(histogram_array)

"""
STEP 2: Pixel mapping lookup table
"""
transform_map = np.floor(255 * chistogram_array).astype(np.uint8)

"""
STEP 3: Transformation
"""
# flatten image array into 1D list
img_list = list(img.flatten())

# transform pixel values to equalize
eq_img_list = [transform_map[p] for p in img_list]

# reshape and write back into img_array
eq_img_array = np.reshape(np.asarray(eq_img_list), img.shape)

plt.imshow(eq_img_array)
plt.show()

windowsize_r = 8
windowsize_c = 8
for r in range(0,img.shape[0] - windowsize_r, windowsize_r):
    for c in range(0,img.shape[1] - windowsize_c, windowsize_c):
        window = img[r:r+windowsize_r,c:c+windowsize_c]
        histogram_array = np.bincount(window.flatten(), minlength=256)

        # normalize
        num_pixels = np.sum(histogram_array)
        histogram_array = histogram_array / num_pixels

        # normalized cumulative histogram
        chistogram_array = np.cumsum(histogram_array)

        """
        STEP 2: Pixel mapping lookup table
        """
        transform_map = np.floor(255 * chistogram_array).astype(np.uint8)

        """
        STEP 3: Transformation
        """
        # flatten image array into 1D list
        img_list = list(window.flatten())

        # transform pixel values to equalize
        eq_img_list = [transform_map[p] for p in img_list]

        # reshape and write back into img_array
        eq_img_array = np.reshape(np.asarray(eq_img_list), window.shape)
        img[r:r + windowsize_r, c:c + windowsize_c]=eq_img_array
        ######################################
        # WRITE EQUALIZED IMAGE TO FILE
        ######################################
        # convert NumPy array to pillow Image and write to file
plt.imshow(img)
plt.title("equalized gray_scale")
plt.show()

