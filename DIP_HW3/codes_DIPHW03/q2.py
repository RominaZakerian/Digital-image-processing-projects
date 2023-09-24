import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt
from scipy import fftpack
from skimage.color import rgb2lab
from scipy.ndimage import gaussian_filter
from matplotlib.colors import hsv_to_rgb
import PIL.Image as P
from matplotlib import image
# # Read image
img1 = cv.imread('donald_x-ray.png', 0)

# Fast Fourier transform algorithm to obtain frequency distribution
f1 = np.fft.fft2(img1)

# The default result center point position is in the upper left corner,
# Call fftshift() function to shift to the middle position
fshift1 = np.fft.fftshift(f1)

# fft result is a complex number, its absolute value result is amplitude
fimg1 = np.log(np.abs(fshift1))

# Show results
plt.subplot(121), plt.imshow(img1, 'gray'), plt.title('Original image')
plt.axis('off')
plt.subplot(122), plt.imshow(fimg1, 'gray'), plt.title('Image spectrum')
plt.axis('off')
plt.show()
# plt.figure()

img2 = cv.imread('donald_ct_scan.png', 0)

# Fast Fourier transform algorithm to obtain frequency distribution
f2 = np.fft.fft2(img2)

# The default result center point position is in the upper left corner,
# Call fftshift() function to shift to the middle position
fshift2 = np.fft.fftshift(f2)

# fft result is a complex number, its absolute value result is amplitude
fimg2 = np.log(np.abs(fshift2))

# Show results
plt.subplot(121), plt.imshow(img2, 'gray'), plt.title('Original Image')
plt.axis('off')
plt.subplot(122), plt.imshow(fimg2, 'gray'), plt.title('Image spectrum')
plt.axis('off')
# plt.show()
# plt.figure()

########## part b ##########

M, N = img1.shape
K = 40
fimg1[M // 2 - K: M // 2 + K, N // 2 - K: N // 2 + K] = 0

# Find all peaks higher than the 98th percentile
peaks = fimg1 < np.percentile(fimg1, 98)

# Shift the peaks back to align with the original spectrum
peaks = np.fft.ifftshift(peaks)

# Make a copy of the original (complex) spectrum
F_dim = f1.copy()

# Set those peak coefficients to zero
F_dim = F_dim * peaks.astype(int)

# Do the inverse Fourier transform to get back to an image.
# Since we started with a real image, we only look at the real part of
# the output.
image_filtered = np.real(np.fft.ifft2(F_dim))

f, (ax0, ax1,ax2) = plt.subplots(1, 3, figsize=(4.8, 7))
ax0.imshow(np.log10(1 + np.abs(np.fft.fftshift(peaks))), cmap='gray')
ax0.set_title('peaks')

ax1.imshow(np.log10(1 + np.abs(np.fft.fftshift(F_dim))), cmap='gray')
ax1.set_title('spectrum of filtered image')

ax2.imshow(image_filtered, cmap='gray')
ax2.set_title('Reconstructed image')
# plt.figure()


def _local_median(img, x, y, k):
    """
    Computes median for k-neighborhood of img[x,y]
    """
    flat = img[x-k : x+k+1, y-k : y+k+1].flatten()
    flat.sort()
    return flat[len(flat)//2]


def median(img, k=3):
    """
    Changes every pixel to the median of its neighboors
    """
    res = np.copy(img)

    for x in range(img.shape[0]):
        for y in range(img.shape[1]):
            if (x-k >= 0 and x+k < img.shape[0]) and \
                    (y-k >= 0 and y+k < img.shape[1]):
                res[x, y] = _local_median(img, x, y, k)

    return res


median_result=median(img2)

filter_result2 = np.fft.fft2(median_result)

# The default result center point position is in the upper left corner,
# Call fftshift() function to shift to the middle position
fshift_result2 = np.fft.fftshift(filter_result2)

# fft result is a complex number, its absolute value result is amplitude
fimg_result2 = np.log(np.abs(fshift_result2))

image_filtered = np.real(np.fft.ifft2(fimg_result2))

# Show results
f_, (ax0, ax1) = plt.subplots(1, 2, figsize=(4.8, 7))
# ax0.imshow(np.log10(1 + np.abs(np.fft.fftshift(peaks))), cmap='gray')
# ax0.set_title('peaks')

ax0.imshow(fimg_result2, cmap='gray')
ax0.set_title('spectrum of filtered image')

ax1.imshow(median_result, cmap='gray')
ax1.set_title('Reconstructed image')
# plt.figure()
plt.show()
############ part c ##############


img3 = cv.imread('donald_childhood.png', 0)

# Fast Fourier transform algorithm to obtain frequency distribution
f3 = np.fft.fft2(img3)

# The default result center point position is in the upper left corner,
# Call fftshift() function to shift to the middle position
fshift3 = np.fft.fftshift(f3)

# fft result is a complex number, its absolute value result is amplitude
fimg3 = np.log(np.abs(fshift3))

# plt.figure()
plt.subplot(121), plt.imshow(img3, 'gray'), plt.title('Original image')
plt.axis('off')

plt.subplot(122), plt.imshow(fimg3, 'gray'), plt.title('Image spectrum')
plt.axis('off')
# plt.show()

result = gaussian_filter(img3, sigma=5)

f3 = np.fft.fft2(result)

# The default result center point position is in the upper left corner,
# Call fftshift() function to shift to the middle position
fshift3 = np.fft.fftshift(f3)

# fft result is a complex number, its absolute value result is amplitude
fimg3 = np.log(np.abs(fshift3))

#
f_1, (ax0, ax1) = plt.subplots(1, 2, figsize=(4.8, 7))
ax0.imshow(fimg3, cmap='gray')
ax0.set_title('Spectrum after suppression')

ax1.imshow(result, cmap='gray')
ax1.set_title('Reconstructed image')
plt.show()

####

img4 = cv.imread('donald_graduation.png', 0)

# Fast Fourier transform algorithm to obtain frequency distribution
f4 = np.fft.fft2(img4)

# The default result center point position is in the upper left corner,
# Call fftshift() function to shift to the middle position
fshift4 = np.fft.fftshift(f4)

# fft result is a complex number, its absolute value result is amplitude
fimg4 = np.log(np.abs(fshift4))

# plt.figure()
plt.subplot(121), plt.imshow(img4, 'gray'), plt.title('Original image')
plt.axis('off')

plt.subplot(122), plt.imshow(fimg4, 'gray'), plt.title('Image spectrum')
plt.axis('off')
# plt.show()

result = gaussian_filter(img4, sigma=5)

f4 = np.fft.fft2(result)

# The default result center point position is in the upper left corner,
# Call fftshift() function to shift to the middle position
fshift4 = np.fft.fftshift(f4)

# fft result is a complex number, its absolute value result is amplitude
fimg4 = np.log(np.abs(fshift4))

f_2, (ax0, ax1) = plt.subplots(1, 2, figsize=(4.8, 7))
ax0.imshow(fimg4, cmap='gray')
ax0.set_title('Spectrum after suppression')

ax1.imshow(result, cmap='gray')
ax1.set_title('Reconstructed image')
plt.show()


######### part d ##########
imgrgb1 = cv.imread('donald_childhood.png')
red = imgrgb1[:, :, 0]
green = imgrgb1[:, :, 1]
blue = imgrgb1[:, :, 2]

reconstructed=np.empty((imgrgb1.shape))
f_red = np.fft.fft2(red)
fshift_red = np.fft.fftshift(f_red)

# fft result is a complex number, its absolute value result is amplitude
fimg_red = np.log(np.abs(fshift_red))

# plt.figure()
plt.subplot(121), plt.imshow(red), plt.title('red channel')
plt.axis('off')

plt.subplot(122), plt.imshow(fimg_red, 'gray'), plt.title('red channel spectrum')
plt.axis('off')

# plt.show()
result = gaussian_filter(red, sigma=5)

reconstructed[:,:,0]=result
f4 = np.fft.fft2(result)

fshift4 = np.fft.fftshift(f4)

# fft result is a complex number, its absolute value result is amplitude
fimg4 = np.log(np.abs(fshift4))

f_2, (ax0, ax1) = plt.subplots(1, 2, figsize=(4.8, 7))
ax0.imshow(fimg4, cmap='gray')
ax0.set_title('Spectrum after suppression')

ax1.imshow(result)
ax1.set_title('Reconstructed red channel')
plt.show()

#####green
f_green = np.fft.fft2(green)
fshift_green = np.fft.fftshift(f_green)

# fft result is a complex number, its absolute value result is amplitude
fimg_green = np.log(np.abs(fshift_green))

# plt.figure()
plt.subplot(121), plt.imshow(green), plt.title('green channel')
plt.axis('off')

plt.subplot(122), plt.imshow(fimg_green, 'gray'), plt.title('green channel spectrum')
plt.axis('off')
# plt.show()

result = gaussian_filter(green, sigma=5)
reconstructed[:,:,1]=result

f4 = np.fft.fft2(result)

fshift4 = np.fft.fftshift(f4)

# fft result is a complex number, its absolute value result is amplitude
fimg4 = np.log(np.abs(fshift4))

f_2, (ax0, ax1) = plt.subplots(1, 2, figsize=(4.8, 7))
ax0.imshow(fimg4, cmap='gray')
ax0.set_title('Spectrum after suppression')

ax1.imshow(result)
ax1.set_title('Reconstructed green channel')
plt.show()

#####blue
f_blue = np.fft.fft2(blue)
fshift_blue = np.fft.fftshift(f_blue)

# fft result is a complex number, its absolute value result is amplitude
fimg_blue = np.log(np.abs(fshift_blue))

# plt.figure()
plt.subplot(121), plt.imshow(blue), plt.title('blue channel')
plt.axis('off')

plt.subplot(122), plt.imshow(fimg_blue, 'gray'), plt.title('blue channel spectrum')
plt.axis('off')
plt.show()

result = gaussian_filter(blue, sigma=5)
reconstructed[:,:,2]=result

f4 = np.fft.fft2(result)

fshift4 = np.fft.fftshift(f4)

# fft result is a complex number, its absolute value result is amplitude
fimg4 = np.log(np.abs(fshift4))

f_2, (ax0, ax1) = plt.subplots(1, 2, figsize=(4.8, 7))
ax0.imshow(fimg4, cmap='gray')
ax0.set_title('Spectrum after suppression')

ax1.imshow(result)
ax1.set_title('Reconstructed blue channel')
# plt.show()

##### RGB2
imgrgb2 = cv.imread('donald_graduation.png')
red = imgrgb2[:, :, 0]
green = imgrgb2[:, :, 1]
blue = imgrgb2[:, :, 2]

reconstructed1=np.empty((imgrgb2.shape))

f_red = np.fft.fft2(red)
fshift_red = np.fft.fftshift(f_red)

# fft result is a complex number, its absolute value result is amplitude
fimg_red = np.log(np.abs(fshift_red))

# plt.figure()
plt.subplot(121), plt.imshow(red), plt.title('red channel')
plt.axis('off')

plt.subplot(122), plt.imshow(fimg_red, 'gray'), plt.title('red channel spectrum')
plt.axis('off')
# plt.show()

result = gaussian_filter(red, sigma=5)
reconstructed1[:,:,2]=result

f4 = np.fft.fft2(result)

fshift4 = np.fft.fftshift(f4)

# fft result is a complex number, its absolute value result is amplitude
fimg4 = np.log(np.abs(fshift4))

f_2, (ax0, ax1) = plt.subplots(1, 2, figsize=(4.8, 7))
ax0.imshow(fimg4, cmap='gray')
ax0.set_title('Spectrum after suppression')

ax1.imshow(result)
ax1.set_title('Reconstructed red channel')
plt.show()


#####green
f_green = np.fft.fft2(green)
fshift_green = np.fft.fftshift(f_green)

# fft result is a complex number, its absolute value result is amplitude
fimg_green = np.log(np.abs(fshift_green))

# plt.figure()
plt.subplot(121), plt.imshow(green), plt.title('green channel')
plt.axis('off')

plt.subplot(122), plt.imshow(fimg_green, 'gray'), plt.title('green channel spectrum')
plt.axis('off')
# plt.show()


result = gaussian_filter(green, sigma=5)
reconstructed1[:,:,1]=result

f4 = np.fft.fft2(result)

fshift4 = np.fft.fftshift(f4)

# fft result is a complex number, its absolute value result is amplitude
fimg4 = np.log(np.abs(fshift4))

f_2, (ax0, ax1) = plt.subplots(1, 2, figsize=(4.8, 7))
ax0.imshow(fimg4, cmap='gray')
ax0.set_title('Spectrum after suppression')

ax1.imshow(result)
ax1.set_title('Reconstructed green channel')
plt.show()

#####blue
f_blue = np.fft.fft2(blue)
fshift_blue = np.fft.fftshift(f_blue)

# fft result is a complex number, its absolute value result is amplitude
fimg_blue = np.log(np.abs(fshift_blue))

# plt.figure()
plt.subplot(121), plt.imshow(blue), plt.title('blue channel')
plt.axis('off')

plt.subplot(122), plt.imshow(fimg_blue, 'gray'), plt.title('blue channel spectrum')
plt.axis('off')
# plt.show()

result = gaussian_filter(blue, sigma=5)
reconstructed1[:,:,0]=result

f4 = np.fft.fft2(result)

fshift4 = np.fft.fftshift(f4)

# fft result is a complex number, its absolute value result is amplitude
fimg4 = np.log(np.abs(fshift4))

f_2, (ax0, ax1) = plt.subplots(1, 2, figsize=(4.8, 7))
ax0.imshow(fimg4, cmap='gray')
ax0.set_title('Spectrum after suppression')

ax1.imshow(result)
ax1.set_title('Reconstructed blue channel')
plt.show()

# output_image = cv.merge ( [image_filtered_red, image_filtered_green, image_filtered_blue] )
plt.imshow(reconstructed1.astype(np.uint8))
plt.title("reconstructed image")
plt.show()

imgGray = 0.2989 * reconstructed1[:,:,2] + 0.5870 * reconstructed1[:,:,1] + 0.1140 * reconstructed1[:,:,0]

plt.imshow(imgGray, cmap='gray')
plt.title("reconstructed gray image")
plt.show()

plt.imshow(reconstructed.astype(np.uint8))
plt.title("reconstructed image")
plt.show()

imgGray = 0.2989 * reconstructed[:,:,2] + 0.5870 * reconstructed[:,:,1] + 0.1140 * reconstructed[:,:,0]

plt.imshow(imgGray, cmap='gray')
plt.title("reconstructed gray image")
plt.show()
###############HSV channels

img3_HSV = cv.cvtColor(imgrgb1, cv.COLOR_RGB2HSV)
h, s, v = img3_HSV[:, :, 0], img3_HSV[:, :, 1], img3_HSV[:, :, 2]
reconstructed2=np.empty((img3_HSV.shape))
reconstructed2[:,:,0]=h
reconstructed2[:,:,1]=s
reconstructed2[:,:,2]=v

plt.imshow(reconstructed2.astype(np.uint8))
plt.title("original HSV image")
plt.show()


f_h = np.fft.fft2(h)
fshift_h = np.fft.fftshift(f_h)

fimg_h = np.log(np.abs(fshift_h))

plt.subplot(121), plt.imshow(h), plt.title('h channel')
plt.axis('off')
plt.subplot(122), plt.imshow(fimg_h, 'gray'), plt.title('h channel spectrum')
plt.axis('off')
# plt.show()

result = gaussian_filter(h, sigma=5)
reconstructed2[:,:,0]=result

f4 = np.fft.fft2(result)

fshift4 = np.fft.fftshift(f4)

# fft result is a complex number, its absolute value result is amplitude
fimg4 = np.log(np.abs(fshift4))

f_2, (ax0, ax1) = plt.subplots(1, 2, figsize=(4.8, 7))
ax0.imshow(fimg4, cmap='gray')
ax0.set_title('Spectrum after suppression')

ax1.imshow(result)
ax1.set_title('Reconstructed h channel')
plt.show()# plt.show()

#####S channel
f_s = np.fft.fft2(s)
fshift_s = np.fft.fftshift(f_s)

# fft result is a complex number, its absolute value result is amplitude
fimg_s = np.log(np.abs(fshift_s))

# plt.figure()
plt.subplot(121), plt.imshow(s), plt.title('s channel')
plt.axis('off')

plt.subplot(122), plt.imshow(fimg_s, 'gray'), plt.title('s channel spectrum')
plt.axis('off')
# plt.show()


result = gaussian_filter(s, sigma=5)
reconstructed2[:,:,1]=result

f4 = np.fft.fft2(result)

fshift4 = np.fft.fftshift(f4)

# fft result is a complex number, its absolute value result is amplitude
fimg4 = np.log(np.abs(fshift4))

f_2, (ax0, ax1) = plt.subplots(1, 2, figsize=(4.8, 7))
ax0.imshow(fimg4, cmap='gray')
ax0.set_title('Spectrum after suppression')

ax1.imshow(result)
ax1.set_title('Reconstructed s channel')
plt.show()# plt.show()

##### V channel
f_v = np.fft.fft2(v)
fshift_v = np.fft.fftshift(f_v)

# fft result is a complex number, its absolute value result is amplitude
fimg_v = np.log(np.abs(fshift_v))

# plt.figure()
plt.subplot(121), plt.imshow(v), plt.title('v channel')
plt.axis('off')

plt.subplot(122), plt.imshow(fimg_v, 'gray'), plt.title('v channel spectrum')
plt.axis('off')
# plt.show()


result = gaussian_filter(v, sigma=5)
reconstructed2[:,:,2]=result

f4 = np.fft.fft2(result)

fshift4 = np.fft.fftshift(f4)

# fft result is a complex number, its absolute value result is amplitude
fimg4 = np.log(np.abs(fshift4))

f_2, (ax0, ax1) = plt.subplots(1, 2, figsize=(4.8, 7))
ax0.imshow(fimg4, cmap='gray')
ax0.set_title('Spectrum after suppression')

ax1.imshow(result)
ax1.set_title('Reconstructed v channel')
plt.show()# plt.show()

plt.imshow(reconstructed2.astype(np.uint8))
plt.title("reconstructed hsv image")
plt.show()



##### HSV2
img4_HSV = cv.cvtColor(imgrgb2, cv.COLOR_RGB2HSV)
reconstructed3=np.empty((img4_HSV.shape))

# img3_HSV = cv.cvtColor(imgrgb1, cv.COLOR_RGB2HSV)
h, s, v = img4_HSV[:, :, 0], img4_HSV[:, :, 1], img4_HSV[:, :, 2]

reconstructed3[:,:,0]=h
reconstructed3[:,:,1]=s
reconstructed3[:,:,2]=v

plt.imshow(reconstructed3.astype(np.uint8))
plt.title("original HSV image")
plt.show()

f_h = np.fft.fft2(h)
fshift_h = np.fft.fftshift(f_h)

# fft result is a complex number, its absolute value result is amplitude
fimg_h = np.log(np.abs(fshift_h))

plt.figure()
plt.subplot(121), plt.imshow(h), plt.title('h channel')
plt.axis('off')

plt.subplot(122), plt.imshow(fimg_h, 'gray'), plt.title('h channel spectrum')
plt.axis('off')
# plt.show()


result = gaussian_filter(h, sigma=5)
reconstructed3[:,:,0]=result

f4 = np.fft.fft2(result)

fshift4 = np.fft.fftshift(f4)

# fft result is a complex number, its absolute value result is amplitude
fimg4 = np.log(np.abs(fshift4))

f_2, (ax0, ax1) = plt.subplots(1, 2, figsize=(4.8, 7))
ax0.imshow(fimg4, cmap='gray')
ax0.set_title('Spectrum after suppression')

ax1.imshow(result)
ax1.set_title('Reconstructed s channel')
plt.show()
#####S channel
f_s = np.fft.fft2(s)
fshift_s = np.fft.fftshift(f_s)

# fft result is a complex number, its absolute value result is amplitude
fimg_s = np.log(np.abs(fshift_s))

plt.figure()
plt.subplot(121), plt.imshow(s), plt.title('s channel')
plt.axis('off')

plt.subplot(122), plt.imshow(fimg_s, 'gray'), plt.title('s channel spectrum')
plt.axis('off')
# plt.show()

##### V channel
f_v = np.fft.fft2(v)
fshift_v = np.fft.fftshift(f_v)

# fft result is a complex number, its absolute value result is amplitude
fimg_v = np.log(np.abs(fshift_v))

plt.figure()
plt.subplot(121), plt.imshow(v), plt.title('v channel')
plt.axis('off')

plt.subplot(122), plt.imshow(fimg_v, 'gray'), plt.title('v channel spectrum')
plt.axis('off')
plt.show()

result = gaussian_filter(s, sigma=5)
reconstructed3[:,:,1]=result

f4 = np.fft.fft2(result)

fshift4 = np.fft.fftshift(f4)

# fft result is a complex number, its absolute value result is amplitude
fimg4 = np.log(np.abs(fshift4))

f_2, (ax0, ax1) = plt.subplots(1, 2, figsize=(4.8, 7))
ax0.imshow(fimg4, cmap='gray')
ax0.set_title('Spectrum after suppression')

ax1.imshow(result)
ax1.set_title('Reconstructed h channel')
plt.show()

result = gaussian_filter(v, sigma=5)
reconstructed3[:,:,2]=result

f4 = np.fft.fft2(result)

fshift4 = np.fft.fftshift(f4)

# fft result is a complex number, its absolute value result is amplitude
fimg4 = np.log(np.abs(fshift4))

f_2, (ax0, ax1) = plt.subplots(1, 2, figsize=(4.8, 7))
ax0.imshow(fimg4, cmap='gray')
ax0.set_title('Spectrum after suppression')

ax1.imshow(result)
ax1.set_title('Reconstructed v channel')
plt.show()

plt.imshow(reconstructed3.astype(np.uint8))
plt.title("reconstructed hsv image")
plt.show()
####################rgb2ycbr channels
def rgb2ycbcr(im):
    cbcr = np.empty_like(im)
    r = im[:,:,0]
    g = im[:,:,1]
    b = im[:,:,2]
    # Y
    cbcr[:,:,0] = .299 * r + .587 * g + .114 * b
    # Cb
    cbcr[:,:,1] = 128 - .169 * r - .331 * g + .5 * b
    # Cr
    cbcr[:,:,2] = 128 + .5 * r - .419 * g - .081 * b
    return np.uint8(cbcr)

yubcr_image3=rgb2ycbcr(imgrgb1)

y, cb, cr = yubcr_image3[:, :, 0], yubcr_image3[:, :, 1], yubcr_image3[:, :, 2]

f_y = np.fft.fft2(y)
fshift_y = np.fft.fftshift(f_y)

# fft result is a complex number, its absolute value result is amplitude
fimg_y = np.log(np.abs(fshift_y))

# plt.figure()
plt.subplot(121), plt.imshow(y), plt.title('y channel')
plt.axis('off')

plt.subplot(122), plt.imshow(fimg_y, 'gray'), plt.title('y channel spectrum')
plt.axis('off')
# plt.show()

#####Cb channel
f_cb = np.fft.fft2(cb)
fshift_cb = np.fft.fftshift(f_cb)

# fft result is a complex number, its absolute value result is amplitude
fimg_cb = np.log(np.abs(fshift_cb))

plt.figure()
plt.subplot(121), plt.imshow(cb), plt.title('cb channel')
plt.axis('off')

plt.subplot(122), plt.imshow(fimg_cb, 'gray'), plt.title('cb channel spectrum')
plt.axis('off')
# plt.show()

##### cr channel
f_cr = np.fft.fft2(cr)
fshift_cr = np.fft.fftshift(f_cr)

# fft result is a complex number, its absolute value result is amplitude
fimg_cr = np.log(np.abs(fshift_cr))

plt.figure()
plt.subplot(121), plt.imshow(cr), plt.title('cr channel')
plt.axis('off')

plt.subplot(122), plt.imshow(fimg_cr, 'gray'), plt.title('cr channel spectrum')
plt.axis('off')
plt.show()

#### YUCBR2
yubcr_image4=rgb2ycbcr(imgrgb2)

y, cb, cr = yubcr_image4[:, :, 0], yubcr_image4[:, :, 1], yubcr_image4[:, :, 2]

f_y = np.fft.fft2(y)
fshift_y = np.fft.fftshift(f_y)

# fft result is a complex number, its absolute value result is amplitude
fimg_y = np.log(np.abs(fshift_y))

plt.figure()
plt.subplot(121), plt.imshow(y), plt.title('y channel')
plt.axis('off')

plt.subplot(122), plt.imshow(fimg_y, 'gray'), plt.title('y channel spectrum')
plt.axis('off')
# plt.show()

#####Cb channel
f_cb = np.fft.fft2(cb)
fshift_cb = np.fft.fftshift(f_cb)

# fft result is a complex number, its absolute value result is amplitude
fimg_cb = np.log(np.abs(fshift_cb))

plt.figure()
plt.subplot(121), plt.imshow(cb), plt.title('cb channel')
plt.axis('off')

plt.subplot(122), plt.imshow(fimg_cb, 'gray'), plt.title('cb channel spectrum')
plt.axis('off')
# plt.show()

##### cr channel
f_cr = np.fft.fft2(cr)
fshift_cr = np.fft.fftshift(f_cr)

# fft result is a complex number, its absolute value result is amplitude
fimg_cr = np.log(np.abs(fshift_cr))

plt.figure()
plt.subplot(121), plt.imshow(cr), plt.title('cr channel')
plt.axis('off')

plt.subplot(122), plt.imshow(fimg_cr, 'gray'), plt.title('cr channel spectrum')
plt.axis('off')
plt.show()


####RGB to lab

imglab1 = rgb2lab(imgrgb1)

L, A, B = imglab1[:, :, 0], imglab1[:, :, 1], imglab1[:, :, 2]

f_L = np.fft.fft2(L)
fshift_L = np.fft.fftshift(f_L)

# fft result is a complex number, its absolute value result is amplitude
fimg_L = np.log(np.abs(fshift_L))

plt.figure()
plt.subplot(121), plt.imshow(L), plt.title('L channel')
plt.axis('off')

plt.subplot(122), plt.imshow(fimg_L, 'gray'), plt.title('L channel spectrum')
plt.axis('off')
# plt.show()

#####L channel
f_A = np.fft.fft2(A)
fshift_A = np.fft.fftshift(f_A)

# fft result is a complex number, its absolute value result is amplitude
fimg_A = np.log(np.abs(fshift_A))

plt.figure()
plt.subplot(121), plt.imshow(A), plt.title('A channel')
plt.axis('off')

plt.subplot(122), plt.imshow(fimg_A, 'gray'), plt.title('A channel spectrum')
plt.axis('off')
# plt.show()

##### B channel
f_B = np.fft.fft2(B)
fshift_B = np.fft.fftshift(f_B)

# fft result is a complex number, its absolute value result is amplitude
fimg_B = np.log(np.abs(fshift_B))

plt.figure()
plt.subplot(121), plt.imshow(B), plt.title('B channel')
plt.axis('off')

plt.subplot(122), plt.imshow(fimg_B, 'gray'), plt.title('B channel spectrum')
plt.axis('off')
plt.show()


imglab2 = rgb2lab(imgrgb2)

L, A, B = imglab2[:, :, 0], imglab2[:, :, 1], imglab2[:, :, 2]

f_L = np.fft.fft2(L)
fshift_L = np.fft.fftshift(f_L)

# fft result is a complex number, its absolute value result is amplitude
fimg_L = np.log(np.abs(fshift_L))

plt.figure()
plt.subplot(121), plt.imshow(L), plt.title('L channel')
plt.axis('off')

plt.subplot(122), plt.imshow(fimg_L, 'gray'), plt.title('L channel spectrum')
plt.axis('off')
# plt.show()

#####L channel
f_A = np.fft.fft2(A)
fshift_A = np.fft.fftshift(f_A)

# fft result is a complex number, its absolute value result is amplitude
fimg_A = np.log(np.abs(fshift_A))

plt.figure()
plt.subplot(121), plt.imshow(A), plt.title('A channel')
plt.axis('off')

plt.subplot(122), plt.imshow(fimg_A, 'gray'), plt.title('A channel spectrum')
plt.axis('off')
# plt.show()

##### B channel
f_B = np.fft.fft2(B)
fshift_B = np.fft.fftshift(f_B)

# fft result is a complex number, its absolute value result is amplitude
fimg_B = np.log(np.abs(fshift_B))

plt.figure()
plt.subplot(121), plt.imshow(B), plt.title('B channel')
plt.axis('off')

plt.subplot(122), plt.imshow(fimg_B, 'gray'), plt.title('B channel spectrum')
plt.axis('off')
plt.show()

################ part e
