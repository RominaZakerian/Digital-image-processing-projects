import numpy as np
import cv2
from scipy import ndimage
from PIL import Image
import math
# importing library for plotting
from matplotlib import pyplot as plt

# # reads an input image
# img = cv2.imread('nasir_and_wives.png',0)
#
# plt.imshow(img,'gray')
# plt.title("original image")
# plt.figure()
#
# # find frequency of pixels in range 0-255
# histr = cv2.calcHist([img], [0], None, [256], [0, 256])
#
# # show the plotting graph of an image
# plt.plot(histr)
# plt.figure()
#
#
# def H_mean(img, kernel_size):
#     G_mean_img = np.zeros(img.shape)
#     # print(G_mean_img[0][0])
#
#     # print(img)
#     k = int((kernel_size - 1) / 2)
#     # print(k)
#     for i in range(img.shape[0]):
#         for j in range(img.shape[1]):
#             if i < k or i > (img.shape[0] - k - 1) or j < k or j > (img.shape[1] - k - 1):
#                 G_mean_img[i][j] = img[i][j]
#             else:
#                 for n in range(kernel_size):
#                     for m in range(kernel_size):
#                         if img[i - k + n][j - k + m] == 0:
#                             G_mean_img[i][j] = 0
#                             break
#                         else:
#                             G_mean_img[i][j] += 1 / np.float(img[i - k + n][j - k + m])
#                     else:
#                         continue
#                     break
#                 if G_mean_img[i][j] != 0:
#                     G_mean_img[i][j] = (kernel_size * kernel_size) / G_mean_img[i][j]
#
#                 # G_mean_img[i][j]=1/9*(img[i-1][j-1]+img[i-1][j]+img[i-1][j+1]+img[i][j-1]+img[i][j]+img[i][j+1]+img[i+1][j-1]+img[i+1][j]+img[i+1][j+1])
#     G_mean_img = np.uint8(G_mean_img)
#
#     return G_mean_img
#
# # plt.imshow(img,'gray')
# # plt.figure()
# result=H_mean(img,5)
# # result = cv2.cvtColor(result.astype(np.uint8),cv2.COLOR_GRAY2BGR)
# plt.imshow(result,'gray')
# plt.title("after harmonic mean")
# # plt.figure()

def convolution(oldimage, kernel):
    # image = Image.fromarray(image, 'RGB')
    image_h = oldimage.shape[0]
    image_w = oldimage.shape[1]

    kernel_h = kernel.shape[0]
    kernel_w = kernel.shape[1]

    if (len(oldimage.shape) == 3):
        image_pad = np.pad(oldimage, pad_width=((kernel_h // 2, kernel_h // 2), (kernel_w // 2,kernel_w // 2), (0, 0)), mode = 'constant',constant_values = 0).astype(np.float32)
    elif (len(oldimage.shape) == 2):
        image_pad = np.pad(oldimage, pad_width=((kernel_h // 2, kernel_h // 2), (kernel_w // 2,kernel_w // 2)), mode = 'constant', constant_values = 0).astype(np.float32)


    h = kernel_h // 2
    w = kernel_w // 2

    image_conv = np.zeros(image_pad.shape)

    for i in range(h, image_pad.shape[0] - h):
        for j in range(w, image_pad.shape[1] - w):
            # sum = 0
            x = image_pad[i - h:i - h + kernel_h, j - w:j - w + kernel_w]
            x = x.flatten() * kernel.flatten()
            image_conv[i][j] = x.sum()
    h_end = -h
    w_end = -w

    if (h == 0):
        return image_conv[h:, w:w_end]
    if (w == 0):
        return image_conv[h:h_end, w:]
    return image_conv[h:h_end, w:w_end]


def GaussianBlurImage(image, sigma):
    # image = imread(image)
    # image =  cv2.imread('nasir_and_wives.png',0)
    image = np.asarray(image)
    # print(image)
    filter_size = 2 * int(4 * sigma + 0.5) + 1
    gaussian_filter = np.zeros((filter_size, filter_size), np.float32)
    m = filter_size // 2
    n = filter_size // 2

    for x in range(-m, m + 1):
        for y in range(-n, n + 1):
            x1 = 2 * np.pi * (sigma ** 2)
            x2 = np.exp(-(x ** 2 + y ** 2) / (2 * sigma ** 2))
            gaussian_filter[x + m, y + n] = (1 / x1) * x2

    im_filtered = np.zeros_like(image, dtype=np.float32)
    # for c in range(2):
    # im_filtered[:, :]=ndimage.convolve(image[:, :], gaussian_filter, mode='constant', cval=0.0)
    im_filtered[:, :] = convolution(image[:, :], gaussian_filter)
    return (im_filtered.astype(np.uint8))

# result=GaussianBlurImage(img,2)
#
# plt.imshow(result,'gray')
# plt.title("after gaussian blur")
# plt.figure()

# kernel = np.array([[-1,-1,-1], [-1, 9,-1],[-1,-1,-1]])
# result = convolution(result,kernel)
# plt.imshow(result,'gray')
# # plt.figure()
# plt.title("after sharpening filter")

# plt.show()

# ########################picture 2###################################

# img = cv2.imread('nasir_and_dentist.png',0)
#
# plt.imshow(img,'gray')
# plt.title("original image")
# plt.figure()
#
# # find frequency of pixels in range 0-255
# histr = cv2.calcHist([img], [0], None, [256], [0, 256])
#
# # show the plotting graph of an image
# plt.plot(histr)
# plt.figure()
#
# # img_noisy1 = cv2.imread('sample.png', 0)
#
# # Obtain the number of rows and columns
# # of the image
# m, n = img.shape
#
# # Traverse the image. For every 3X3 area,
# # find the median of the pixels and
# # replace the ceter pixel by the median
# img_new1 = np.zeros([m, n])
#
# for i in range(1, m - 1):
#     for j in range(1, n - 1):
#         temp = [img[i - 1, j - 1],
#                 img[i - 1, j],
#                 img[i - 1, j + 1],
#                 img[i, j - 1],
#                 img[i, j],
#                 img[i, j + 1],
#                 img[i + 1, j - 1],
#                 img[i + 1, j],
#                 img[i + 1, j + 1]]
#
#         temp = sorted(temp)
#         img_new1[i, j] = temp[4]
#
# img_new1 = img_new1.astype(np.uint8)
# # cv2.imwrite('new_median_filtered.png', img_new1)
#
# # plt.imshow(img,'gray')
# # plt.figure()
# plt.imshow(img_new1,'gray')
# plt.title("after median filter")
# plt.figure()
#
# histr = cv2.calcHist([img_new1], [0], None, [256], [0, 256])
#
# # show the plotting graph of an image
# plt.plot(histr)
# plt.figure()
#
# img_new2 = np.zeros([m, n])
#
# for i in range(1, m - 1):
#     for j in range(1, n - 1):
#         temp = [img_new1[i - 1, j - 1],
#                 img_new1[i - 1, j],
#                 img_new1[i - 1, j + 1],
#                 img_new1[i, j - 1],
#                 img_new1[i, j],
#                 img_new1[i, j + 1],
#                 img_new1[i + 1, j - 1],
#                 img_new1[i + 1, j],
#                 img_new1[i + 1, j + 1]]
#
#         temp = sorted(temp)
#         img_new2[i, j] = temp[4]
#
# img_new2 = img_new2.astype(np.uint8)
# # plt.imshow(img,'gray')
# # plt.figure()
# # plt.imshow(img_new1,'gray')
# # plt.figure()
# plt.imshow(img_new2,'gray')
# plt.title("after second median filter")
# plt.figure()
# def H_mean(img, kernel_size):
#     G_mean_img = np.zeros(img.shape)
#     # print(G_mean_img[0][0])
#
#     # print(img)
#     k = int((kernel_size - 1) / 2)
#     # print(k)
#     for i in range(img.shape[0]):
#         for j in range(img.shape[1]):
#             if i < k or i > (img.shape[0] - k - 1) or j < k or j > (img.shape[1] - k - 1):
#                 G_mean_img[i][j] = img[i][j]
#             else:
#                 for n in range(kernel_size):
#                     for m in range(kernel_size):
#                         if img[i - k + n][j - k + m] == 0:
#                             G_mean_img[i][j] = 0
#                             break
#                         else:
#                             G_mean_img[i][j] += 1 / np.float(img[i - k + n][j - k + m])
#                     else:
#                         continue
#                     break
#                 if G_mean_img[i][j] != 0:
#                     G_mean_img[i][j] = (kernel_size * kernel_size) / G_mean_img[i][j]
#
#                 # G_mean_img[i][j]=1/9*(img[i-1][j-1]+img[i-1][j]+img[i-1][j+1]+img[i][j-1]+img[i][j]+img[i][j+1]+img[i+1][j-1]+img[i+1][j]+img[i+1][j+1])
#     G_mean_img = np.uint8(G_mean_img)
#
#     return G_mean_img
#
# # plt.imshow(img,'gray')
# # plt.figure()
# # plt.imshow(img_new1,'gray')
# # plt.figure()
# result=H_mean(img_new1,3)
# plt.imshow(result,'gray')
# plt.title("after harmonic filter")
# # plt.figure()
#
#
# plt.show()

# ########################picture 3###################################
#
# img = cv2.imread('nasir_smoking_hookah.png',0)
#
# plt.imshow(img,'gray')
# plt.title("original image")
# plt.figure()
#
# # find frequency of pixels in range 0-255
# histr = cv2.calcHist([img], [0], None, [256], [0, 256])
#
# # show the plotting graph of an image
# plt.plot(histr)
# plt.figure()
#
# # img_noisy1 = cv2.imread('sample.png', 0)
#
# # Obtain the number of rows and columns
# # of the image
# m, n = img.shape
#
# # Traverse the image. For every 3X3 area,
# # find the median of the pixels and
# # replace the ceter pixel by the median
# img_new1 = np.zeros([m, n])
#
# for i in range(1, m - 1):
#     for j in range(1, n - 1):
#         temp = [img[i - 1, j - 1],
#                 img[i - 1, j],
#                 img[i - 1, j + 1],
#                 img[i, j - 1],
#                 img[i, j],
#                 img[i, j + 1],
#                 img[i + 1, j - 1],
#                 img[i + 1, j],
#                 img[i + 1, j + 1]]
#
#         temp = sorted(temp)
#         img_new1[i, j] = temp[4]
#
# img_new1 = img_new1.astype(np.uint8)
# # cv2.imwrite('new_median_filtered.png', img_new1)
#
# # plt.imshow(img,'gray')
# # plt.figure()
# plt.imshow(img_new1,'gray')
# plt.title("after median filter")
# plt.figure()
#
# img_new2 = np.zeros([m, n])
#
# for i in range(1, m - 1):
#     for j in range(1, n - 1):
#         temp = [img_new1[i - 1, j - 1],
#                 img_new1[i - 1, j],
#                 img_new1[i - 1, j + 1],
#                 img_new1[i, j - 1],
#                 img_new1[i, j],
#                 img_new1[i, j + 1],
#                 img_new1[i + 1, j - 1],
#                 img_new1[i + 1, j],
#                 img_new1[i + 1, j + 1]]
#
#         temp = sorted(temp)
#         img_new2[i, j] = temp[4]
#
# img_new2 = img_new2.astype(np.uint8)
# # cv2.imwrite('new_median_filtered.png', img_new1)
#
# # plt.imshow(img,'gray')
# # plt.figure()
# plt.imshow(img_new2,'gray')
# plt.title("after second median filter")
# plt.figure()
#
# histr = cv2.calcHist([img_new1], [0], None, [256], [0, 256])
#
# # show the plotting graph of an image
# plt.plot(histr)
# plt.show()

########################picture 4###################################

img = cv2.imread('nasir_receiving_pachekhari.png',0)

print(img.shape)
plt.imshow(img,'gray')
plt.title("original image")
plt.figure()

# find frequency of pixels in range 0-255
histr = cv2.calcHist([img], [0], None, [256], [0, 256])

# show the plotting graph of an image
plt.plot(histr)
plt.figure()

f1 = np.fft.fft2(img)

# The default result center point position is in the upper left corner,
# Call fftshift() function to shift to the middle position
fshift1 = np.fft.fftshift(f1)

# fft result is a complex number, its absolute value result is amplitude
fimg1 = np.log(np.abs(fshift1))

# Show results
plt.subplot(121), plt.imshow(img, 'gray'), plt.title('Original image')
plt.axis('off')
plt.subplot(122), plt.imshow(fimg1, 'gray'), plt.title('Image spectrum')
plt.axis('off')
plt.figure()


spec_orig = np.fft.fft2(img)
spec_img = np.fft.fftshift(spec_orig)

for j in range(535,735):
    for n in range(285,295):
        spec_img[n,j] = 0

    for n in range(585, 595):
        spec_img[n,j] = 0

for j in range(615, 665):
    for n in range(135, 145):
        spec_img[n, j] = 0

    for n in range(730, 740):
        spec_img[n, j] = 0

for j in range(435, 445):
    for n in range(0, 570):
        spec_img[j, n] = 0

    for n in range(710, 1279):
        spec_img[j, n] = 0


# ptnfx = np.real(np.fft.ifft2(np.fft.ifftshift(spec_img)))
fimg1 = np.log(np.abs(spec_img))
plt.imshow(fimg1,'gray')
plt.figure()
# %// Change
# plt.imshow(img,'gray')
# plt.figure()
ptnfx = np.real(np.fft.ifft2(np.fft.ifftshift(spec_img)))
plt.imshow(ptnfx,'gray')
plt.title("after notch filter")
# plt.figure()
# kernel = np.array([[-1,-1,-1], [-1, 9,-1],[-1,-1,-1]])
# result = cv2.filter2D(ptnfx,-1,kernel)
# plt.imshow(result,'gray')
# # plt.figure()
# plt.title("after sharpening filter")

plt.show()
