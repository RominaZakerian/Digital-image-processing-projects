import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from PIL import Image
import cv2
from skimage import io
from skimage import color
from scipy import ndimage
import math
from matplotlib.pyplot import cm
###### picture 1 ###########
img = cv2.imread('quran_1.png')
data=np.asarray(img)
imgplot = plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
print(data)
print(data.shape)
plt.title("original image")
# plt.figure()
# plt.show()

img_gray = io.imread('quran_1.png', as_gray=True)

plt.imshow(img_gray,cmap=cm.gray)

plt.title("gray_image")

#
# img = io.imread('quran_1.png')
# imgGray = color.rgb2gray(img)

histogram, bin_edges = np.histogram(img_gray, bins=256, range=(0, 1))

plt.figure()
plt.title("Grayscale Histogram")
plt.xlabel("grayscale value")
plt.ylabel("pixels")
plt.xlim([0.0, 1.0])  # <- named arguments do not work here

plt.plot(bin_edges[0:-1], histogram)  # <- or here
plt.show()
########## global thresholding ##############
for i in range(img_gray.shape[0]):
    for j in range(img_gray.shape[1]):

        if img_gray[i,j]<0.3:
            img_gray[i,j]=0
        else:
            img_gray[i,j]=1

print(img_gray)
plt.imshow(img_gray,cmap=cm.gray)
plt.title("global_thresholding")
plt.show()

########## adaptive mean thresholding ###########
windowsize_r = 8
windowsize_c = 8

# Crop out the window and calculate the histogram
for r in range(0,img.shape[0] - windowsize_r, windowsize_r):
    for c in range(0,img.shape[1] - windowsize_c, windowsize_c):
        window = img[r:r+windowsize_r,c:c+windowsize_c]
        # print(window)
        # print(window.shape)
        block_means = window.mean(axis=0).mean(axis=0)
        # [3.5, 5.8, 5.5, 4.2]
        print(block_means)
        constant=20
        print("subtracting constant")
        threshold=block_means-30
        print(threshold)

        for i in range(window.shape[0]):
            for j in range(window.shape[1]):
                if (window[i,j]<threshold).all():
                    window[i,j]=[0,0,0]

                else:
                    window[i,j]=[255,255,255]

        img[r:r+windowsize_r,c:c+windowsize_c]=window


plt.imshow(img)
plt.title("Adaptive mean thresholding")
plt.figure()

# plt.show()
#################### adaptive gaussian thresholding ################

# Crop out the window and calculate the histogram
for r in range(0, img.shape[0] - windowsize_r, windowsize_r):
    for c in range(0, img.shape[1] - windowsize_c, windowsize_c):
        window = img[r:r + windowsize_r, c:c + windowsize_c]
        sigma=2

        # block_means = window.mean(axis=0).mean(axis=0)
        # [3.5, 5.8, 5.5, 4.2]
        # print(block_means)

        block_std = 1
        print("this is std",block_std)
        sum_result=0
        for i in range(window.shape[0]):
            for j in range(window.shape[1]):
                exponen=np.exp((-1*((i **2)+(j**2)))/(2* (block_std**2)))
                exponen=exponen/(2*(block_std**2)*math.pi)
                result=window[i,j]*exponen
                sum_result=sum_result+result

                # x = vector(window)
                # gaussi=(window[i,j] - block_means) * -1
                # print(window[i,j])
                # print(block_means)
                # print(gaussi)
                # gaussi=gaussi**2
                # stdev=(block_std**2)
                # stdev= 2 * stdev
                # result=gaussi/stdev
                # print("results",result)
                # c
                # # ** 2) / (2 * (block_std ** 2))
                # print("this result ",result)
                # t=t+result


        print("this is weighted gaussian",sum_result)
        threshold = sum_result - 30
        print(threshold)
        for i in range(window.shape[0]):
            for j in range(window.shape[1]):
                if (window[i,j]<threshold).all():
                    window[i,j]=[0,0,0]

                else:
                    window[i,j]=[255,255,255]

        img[r:r+windowsize_r,c:c+windowsize_c]=window
plt.imshow(img)
plt.title("Adaptive gaussian thresholding")
plt.show()


########## otsu method #################


pixel_number = img_gray.shape[0] * img_gray.shape[1]
mean_weigth = 1.0/pixel_number
his, bins = np.histogram(img_gray, np.array(range(0, 256)))
final_thresh = -1
final_value = -1
for t in bins[1:-1]: # This goes from 1 to 254 uint8 range (Pretty sure wont be those values)
    Wb = np.sum(his[:t]) * mean_weigth
    Wf = np.sum(his[t:]) * mean_weigth

    mub = np.mean(his[:t])
    muf = np.mean(his[t:])

    value = Wb * Wf * (mub - muf) ** 2

    print("Wb", Wb, "Wf", Wf)
    print("t", t, "value", value)

    if value > final_value:
        final_thresh = t
        final_value = value
final_img = img_gray.copy()
print(final_thresh)
final_img[img_gray > final_thresh] = 255
final_img[img_gray < final_thresh] = 0

plt.imshow(final_img,cmap=cm.gray)
plt.title("otsu thresholding")
plt.show()






