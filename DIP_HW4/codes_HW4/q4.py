import numpy as np
import cv2
from scipy.ndimage import maximum_filter, minimum_filter
from matplotlib import pyplot as plt
from pylab import *
from scipy import ndimage, misc
from skimage.restoration import inpaint
from scipy import stats
import mahotas
from PIL import Image, ImageFilter
from skimage import io
from skimage.filters import try_all_threshold
from skimage.color import rgb2gray
from scipy.signal.signaltools import wiener

# # reads an input image
# img = cv2.imread('post-mortem_1.png')
# # mask = cv2.imread('mask.png',0)
# mask1 = cv2.imread('mask4.png',0)
# print(mask1.shape)
# print(img.shape)
# plt.imshow(cv2.cvtColor(img,cv2.COLOR_BGR2RGB))
# plt.title("original image")
# plt.figure()
#
# mask = np.zeros(img.shape[:-1], dtype=bool)
# mask[449:479, 0:649] = 1
# mask[0:100, 0:240] = 1
#
# mask=mask.astype(np.uint8)
#
# image_result = cv2.inpaint(img,mask,3,cv2.INPAINT_TELEA)
#
# plt.imshow(cv2.cvtColor(image_result,cv2.COLOR_BGR2RGB))
# plt.title("after interpolation")
# plt.figure()


def HarmonicMeanOperator(roi):
    roi = roi.astype(np.float64)
    if 0 in roi:
        roi = 0
    else:
        roi = stats.hmean(roi.reshape(-1))
    return roi
def HarmonicMeanAlogrithm(image):
    # Harmonic mean filter
    new_image = np.zeros(image.shape)
    image = cv2.copyMakeBorder(image,1,1,1,1,cv2.BORDER_DEFAULT)
    for i in range(1,image.shape[0]-1):
        for j in range(1,image.shape[1]-1):
            new_image[i-1,j-1] =HarmonicMeanOperator(image[i-1:i+2,j-1:j+2])
    new_image = (new_image-np.min(image))*(255/np.max(image))
    return new_image.astype(np.uint8)
def rgbHarmonicMean(image):
    r,g,b = cv2.split(image)
    r = HarmonicMeanAlogrithm(r)
    g = HarmonicMeanAlogrithm(g)
    b = HarmonicMeanAlogrithm(b)
    return cv2.merge([r,g,b])
#
# #
# # def GeometricMeanOperator(roi):
# #     roi = roi.astype(np.float64)
# #     p = np.prod(roi)
# #     return p ** (1 / (roi.shape[0] * roi.shape[1]))
# # def GeometricMeanAlogrithm(image):
# #     # Geometric mean filter
# #     new_image = np.zeros(image.shape)
# #     image = cv2.copyMakeBorder(image, 1, 1, 1, 1, cv2.BORDER_DEFAULT)
# #     for i in range(1, image.shape[0] - 1):
# #         for j in range(1, image.shape[1] - 1):
# #             new_image[i - 1, j - 1] = GeometricMeanOperator(image[i - 1:i + 2, j - 1:j + 2])
# #     new_image = (new_image - np.min(image)) * (255 / np.max(image))
# #     return new_image.astype(np.uint8)
# # def rgbGemotriccMean(image):
# #     b,g,r = cv2.split(image)
# #     r = GeometricMeanAlogrithm(r)
# #     g = GeometricMeanAlogrithm(g)
# #     b = GeometricMeanAlogrithm(b)
# #     return cv2.merge([b,g,r])
#
# result= cv2.GaussianBlur(image_result,(9,9),0)
#
# plt.imshow(cv2.cvtColor(result,cv2.COLOR_BGR2RGB))
# plt.title("after guassian filter")
# plt.figure()
#
#
# img_hsv = cv2.cvtColor(result, cv2.COLOR_RGB2HSV)
#
# # Histogram equalisation on the V-channel
# img_hsv[:, :, 2] = cv2.equalizeHist(img_hsv[:, :, 2])
#
# # convert image back from HSV to RGB
# result = cv2.cvtColor(img_hsv, cv2.COLOR_HSV2RGB)
#
# plt.imshow(cv2.cvtColor(result,cv2.COLOR_BGR2RGB))
# plt.title("after histogram equalization")
# plt.figure()
#
#
# result = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)
# result=rgbHarmonicMean(result)
# plt.imshow(result)
# plt.title("after harmonic filter")
# plt.figure()
#
# kernel = np.array([[-1,-1,-1], [-1, 9,-1],[-1,-1,-1]])
# sharpened = cv2.filter2D(result, -1, kernel)
# plt.imshow(sharpened)
# # plt.figure()
# plt.title("after sharpening filter")
# # plt.figure()
#
# plt.show()
#
#
# # def midpoint(img):
# #     maxf = maximum_filter(img, (3, 3))
# #     minf = minimum_filter(img, (3, 3))
# #     midpoint = (maxf + minf) / 2
# #     return midpoint
#
# # result=midpoint(image_result)
# # result = mahotas.mean_filter(img[:,:,0], 4)
# # plt.imshow(cv2.cvtColor(result,cv2.COLOR_BGR2RGB))
# # plt.title("after midpoint filter")
# # plt.show()
#
# histr = cv2.calcHist([sharpened], [0], None, [256], [0, 256])
#
# # show the plotting graph of an image
# plt.plot(histr)
# plt.show()


########## second picture ##########################################
#
# img = cv2.imread('post-mortem_2.png')
# # mask = cv2.imread('mask.png',0)
# # mask1 = cv2.imread('mask4.png',0)
# # print(mask1.shape)
# print(img.shape)
# plt.imshow(cv2.cvtColor(img,cv2.COLOR_BGR2RGB))
# plt.title("original image")
# plt.figure()
#
# # histr = cv2.calcHist([img], [0], None, [256], [0, 256])
# #
# # # show the plotting graph of an image
# # plt.plot(histr)
# # plt.show()
#
#
#
# img_hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
# # Histogram equalisation on the V-channel
# img_hsv[:, :, 2] = cv2.equalizeHist(img_hsv[:, :, 2])
#
# # convert image back from HSV to RGB
# result = cv2.cvtColor(img_hsv, cv2.COLOR_HSV2RGB)
#
# plt.imshow(cv2.cvtColor(result,cv2.COLOR_BGR2RGB))
# plt.title("after histogram equalization")
# plt.figure()
#
# img_gray = cv2.cvtColor(result,cv2.COLOR_RGB2GRAY)
# f1 = np.fft.fft2(img_gray)
#
# # The default result center point position is in the upper left corner,
# # Call fftshift() function to shift to the middle position
# fshift1 = np.fft.fftshift(f1)
#
# # fft result is a complex number, its absolute value result is amplitude
# fimg1 = np.log(np.abs(fshift1))
#
# # Show results
# plt.subplot(121), plt.imshow(img_gray, 'gray'), plt.title('Original image')
# plt.axis('off')
# plt.subplot(122), plt.imshow(fimg1, 'gray'), plt.title('Image spectrum')
# plt.axis('off')
# plt.figure()
#
#
# spec_orig = np.fft.fft2(img_gray)
# spec_img = np.fft.fftshift(spec_orig)
#
# for j in range(0,350):
#     for n in range(535,545):
#         spec_img[n,j] = 0
#
#
# for j in range(440, 793):
#     for n in range(535, 545):
#         spec_img[n, j] = 0
#
#
# # ptnfx = np.real(np.fft.ifft2(np.fft.ifftshift(spec_img)))
# fimg1 = np.log(np.abs(spec_img))
# plt.imshow(fimg1,'gray')
# plt.figure()
# # %// Change
#
# result = np.real(np.fft.ifft2(np.fft.ifftshift(spec_img)))
# plt.imshow(result,'gray')
# plt.title("after notch filter")
# plt.figure()
#
# result = cv2.cvtColor(result.astype(np.uint8),cv2.COLOR_GRAY2BGR)
#
# # histr = cv2.calcHist([result], [0], None, [256], [0, 256])
# #
# # # show the plotting graph of an image
# # plt.plot(histr)
# # plt.show()
#
# # result = rgb2gray(result)
#
# result = cv2.medianBlur(result,7)
# plt.imshow(cv2.cvtColor(result,cv2.COLOR_BGR2RGB))
# plt.title("after median filter")
# plt.figure()
# #
#
# result=ndimage.maximum_filter(result, size=3)
#
# plt.imshow(cv2.cvtColor(result,cv2.COLOR_BGR2RGB))
# plt.title("after max filter")
# plt.figure()
#
# #
# result = ndimage.minimum_filter(result, size=3)
# plt.imshow(cv2.cvtColor(result,cv2.COLOR_BGR2RGB))
# plt.title("after min filter")
# plt.figure()
#
#
# #
# # result = cv2.bilateralFilter(result,5,75,75)
# # plt.imshow(cv2.cvtColor(result,cv2.COLOR_BGR2RGB))
# # plt.title("after bilateral filter")
# # plt.figure()
#
#
#
# kernel = np.array([[-1,-1,-1], [-1, 9,-1],[-1,-1,-1]])
# sharpened = cv2.filter2D(result, -1, kernel)
# plt.imshow(sharpened)
# # plt.figure()
# plt.title("after sharpening filter")
# plt.figure()
#
# kernel = np.ones((5,5), np.uint8)
# result = cv2.dilate(result, kernel, iterations=1)
# plt.imshow(result)
# plt.title("after dilation")
# plt.figure()
#
#
#
# kernel = np.ones((5,5), np.uint8)
# result = cv2.erode(result, kernel, iterations=1)
# plt.imshow(result)
# plt.title("after erosion")
# plt.figure()
#
# kernel = np.array([[-1,-1,-1], [-1, 9,-1],[-1,-1,-1]])
# sharpened = cv2.filter2D(result, -1, kernel)
# plt.imshow(sharpened)
# # plt.figure()
# plt.title("after sharpening filter")
# plt.show()

############ third picture ################################################
#
# img = cv2.imread('post-mortem_3.png')
# # mask = cv2.imread('mask.png',0)
# # mask1 = cv2.imread('mask_4_1.png',0)
# # print(mask1.shape)
# mask = np.zeros(img.shape[:-1], dtype=bool)
# mask[0:235, 0:215] = 1
#
# plt.imshow(cv2.cvtColor(img,cv2.COLOR_BGR2RGB))
# plt.title("original image")
# plt.figure()
#
#
# print(img.shape)
#
# plt.imshow(mask,'gray')
# plt.title("mask")
# plt.figure()
#
# mask=mask.astype(np.uint8)
#
# result = cv2.inpaint(img,mask,3,cv2.INPAINT_TELEA)
#
# plt.imshow(cv2.cvtColor(result,cv2.COLOR_BGR2RGB))
# plt.title("after interpolation")
# plt.figure()
#
#
# # result = ndimage.minimum_filter(result, size=5)
# # plt.imshow(cv2.cvtColor(result,cv2.COLOR_BGR2RGB))
# # plt.title("after min filter")
# # plt.figure()
#
# # result = ndimage.maximum_filter(result, size=9)
# # plt.imshow(cv2.cvtColor(result,cv2.COLOR_BGR2RGB))
# # plt.title("after max filter")
# # plt.figure()
#
# # img_hsv = cv2.cvtColor(result, cv2.COLOR_RGB2HSV)
# # # Histogram equalisation on the V-channel
# # img_hsv[:, :, 2] = cv2.equalizeHist(img_hsv[:, :, 2])
# #
# # # convert image back from HSV to RGB
# # result = cv2.cvtColor(img_hsv, cv2.COLOR_HSV2RGB)
# #
# # plt.imshow(cv2.cvtColor(result,cv2.COLOR_BGR2RGB))
# # plt.title("after histogram equalization")
# # plt.figure()
#
# result = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)
# result=rgbHarmonicMean(result)
# plt.imshow(result)
# plt.title("after harmonic filter")
# plt.figure()
#
# # result = cv2.GaussianBlur(result,(9)
# # plt.imshow(cv2.cvtColor(result,cv2.COLOR_BGR2RGB))
# # plt.title("after median filter")
# # plt.figure()
# # #
#
# kernel = np.array([[-1,-1,-1], [-1, 9,-1],[-1,-1,-1]])
# result = cv2.filter2D(result, -1, kernel)
# plt.imshow(result)
# # plt.figure()
# plt.title("after sharpening filter")
# plt.figure()
#
#
# histr = cv2.calcHist([result], [0], None, [256], [0, 256])
#
# # show the plotting graph of an image
# plt.plot(histr)
# # plt.show()
#
#
# plt.show()

############ fourth picture ###########################################

img = cv2.imread('post-mortem_4.png')
# mask = cv2.imread('mask.png',0)
# mask1 = cv2.imread('mask_4_1.png',0)
# print(mask1.shape)
# mask = np.zeros(img.shape[:-1], dtype=bool)

# mask[535:583, 0:300] = 1
print(img.shape)
#
# result = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
# result = wiener(result, (50, 50))
plt.imshow(cv2.cvtColor(img,cv2.COLOR_BGR2RGB))
plt.title("original image")
plt.figure()


img_gray = cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)
f1 = np.fft.fft2(img_gray)

# The default result center point position is in the upper left corner,
# Call fftshift() function to shift to the middle position
fshift1 = np.fft.fftshift(f1)

# fft result is a complex number, its absolute value result is amplitude
fimg1 = np.log(np.abs(fshift1))

# Show results
plt.subplot(121), plt.imshow(img_gray, 'gray'), plt.title('Original image')
plt.axis('off')
plt.subplot(122), plt.imshow(fimg1, 'gray'), plt.title('Image spectrum')
plt.axis('off')
plt.figure()

spec_orig = np.fft.fft2(img_gray)
spec_img = np.fft.fftshift(spec_orig)

for j in range(189,263):
    for n in range(200,217):
        spec_img[n,j] = 0

for j in range(204, 242):
    for n in range(90, 107):
        spec_img[n, j] = 0

for j in range(200, 250):
    for n in range(553, 565):
        spec_img[n, j] = 0

for j in range(188, 258):
    for n in range(439, 455):
        spec_img[n, j] = 0

for j in range(0, 150):
    for n in range(325, 330):
        spec_img[n, j] = 0

for j in range(300, 451):
    for n in range(325, 330):
        spec_img[n, j] = 0

# ptnfx = np.real(np.fft.ifft2(np.fft.ifftshift(spec_img)))
# fimg1 = np.log(np.abs(spec_img))
# plt.imshow(fimg1)
# plt.figure()
result = np.real(np.fft.ifft2(np.fft.ifftshift(spec_img)))

plt.imshow(result,'gray')
plt.title("after notch filter")
plt.figure()


result = cv2.cvtColor(result.astype(np.uint8),cv2.COLOR_GRAY2BGR)
print("s,mvmgbzd",result.shape)
# mask=np.empty(result.shape)
#
#
# for i in range(659):
#     for j in range(452):
#         if result[i,j]>250:
#             mask[i,j]=1
#         else:
#             mask[i,j]=0
#
# mask1=np.empty((659,452))
# for i in range(659):
#     for j in range(452):
#         if mask[i,j]==1:
#             mask1[i,j]=1
#             if i+10<659:
#                 mask1[i+10, j] = 1
#             if i-10>0:
#                 mask1[i-10, j] = 1
#             if j+10<452:
#                 mask1[i, j+10] = 1
#             if j-10>0:
#                 mask1[i, j-10] = 1
#             if i-10>0:
#                 if j-10>0:
#                     mask1[i-10, j-10] = 1
#                 if j+10<452:
#                     mask1[i - 10, j + 10] = 1
#             if i+10<659:
#                 if j-10>0:
#                     mask1[i+10, j-10] = 1
#                 if j+10<452:
#                     mask1[i + 10, j + 10] = 1
#
# mask1[0:79, 0:140] = 1
# # mask1=mask1
# mask1=mask1.astype(np.uint8)
# cv2.imwrite("mask_1.png",mask1)
# plt.imshow(mask1,'gray')
# plt.figure()

final_mask = cv2.imread('final_mask.png',0)
plt.imshow(final_mask,'gray')
plt.title("mask")
plt.figure()

mask1=final_mask.astype(np.uint8)
# result = cv2.cvtColor(result.astype(np.uint8),cv2.COLOR_GRAY2BGR)
result = cv2.inpaint(result,final_mask,3,cv2.INPAINT_TELEA)

plt.imshow(result)
plt.title("after interpolation")
plt.figure()

histr = cv2.calcHist([result], [0], None, [256], [0, 256])

# show the plotting graph of an image
plt.plot(histr)
plt.figure()

result = cv2.medianBlur(result,7)
plt.imshow(cv2.cvtColor(result,cv2.COLOR_BGR2RGB))
plt.title("after median filter")
plt.figure()

# result = ndimage.minimum_filter(result, size=3)
# plt.imshow(cv2.cvtColor(result,cv2.COLOR_BGR2RGB))
# plt.title("after min filter")
# plt.figure()
#
# result = ndimage.maximum_filter(result, size=3)
# plt.imshow(cv2.cvtColor(result,cv2.COLOR_BGR2RGB))
# plt.title("after max filter")
# plt.figure()

result = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)
result=rgbHarmonicMean(result)
plt.imshow(result)
plt.title("after harmonic filter")
plt.figure()

# result = cv2.GaussianBlur(result,(5,5),0)
# plt.imshow(cv2.cvtColor(result,cv2.COLOR_BGR2RGB))
# plt.title("after guassian filter")
# plt.figure()
#
# result = cv2.GaussianBlur(result,(5,5),0)
# plt.imshow(cv2.cvtColor(result,cv2.COLOR_BGR2RGB))
# plt.title("after guassian filter")
# plt.figure()

kernel = np.array([[-1,-1,-1], [-1, 9,-1],[-1,-1,-1]])
result = cv2.filter2D(result, -1, kernel)
plt.imshow(result)
# plt.figure()
plt.title("after sharpening filter")
plt.figure()


histr = cv2.calcHist([result], [0], None, [256], [0, 256])

# show the plotting graph of an image
plt.plot(histr)
# plt.figure()
plt.show()
