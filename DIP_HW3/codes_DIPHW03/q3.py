import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt
from scipy import fftpack
from skimage.color import rgb2lab

img1 = cv.imread('donald_1_aligned.png', 0)

# Fast Fourier transform algorithm to obtain frequency distribution
f1 = np.fft.fft2(img1)

# The default result center point position is in the upper left corner,
# Call fftshift() function to shift to the middle position
fshift1 = np.fft.fftshift(f1)

# fft result is a complex number, its absolute value result is amplitude
fimg1 = np.log(np.abs(fshift1))

# Show results
plt.figure(figsize=(6.4*5,4.8*5),constrained_layout=False)
plt.subplot(131), plt.imshow(img1, 'gray'), plt.title('aligned image')
plt.axis('off')
plt.subplot(132), plt.imshow(fimg1, 'gray'), plt.title('amplitude of fourier transform')
plt.axis('off')
# plt.show()


### guassian filters

def distance(points1,points2):
    return np.sqrt((points1[0]-points2[0])**2 + (points1[1]-points2[1])**2)

def guassianLP(D0,imgshape):
    base=np.zeros(imgshape[:2])
    rows,cols=imgshape[:2]
    center=(rows/2,cols/2)
    for x in range(cols):
        for y in range(rows):
            base[y,x] = np.exp(((-distance((y,x),center)**2)/(2*(D0**2))))

    return base

def guassianHP(D0,imgshape):
    base = np.zeros(imgshape[:2])
    rows, cols = imgshape[:2]
    center = (rows / 2, cols / 2)
    for x in range(cols):
        for y in range(rows):
            base[y, x] = 1 - np.exp(((-distance((y, x), center) ** 2) / (2 * (D0 ** 2))))

    return base

LowPassCenter=fshift1*guassianLP(90,img1.shape)
LowPass=np.fft.ifftshift(LowPassCenter)
inverse_LowPass=np.fft.ifft2(LowPass)

plt.subplot(133), plt.imshow(np.abs(inverse_LowPass), 'gray'), plt.title('Guassian LowPass')
plt.axis('off')
plt.show()

plt.imshow(np.log(1+np.abs(LowPassCenter)),cmap='gray')
plt.title("amplitude of FT for filtered image")
plt.show()

####picture 2


img2 = cv.imread('donald_6_aligned.png', 0)

# Fast Fourier transform algorithm to obtain frequency distribution
f2 = np.fft.fft2(img2)

# The default result center point position is in the upper left corner,
# Call fftshift() function to shift to the middle position
fshift2 = np.fft.fftshift(f2)

# fft result is a complex number, its absolute value result is amplitude
fimg2 = np.log(np.abs(fshift2))

# Show results
plt.figure(figsize=(6.4*5,4.8*5),constrained_layout=False)
plt.subplot(131), plt.imshow(img2, 'gray'), plt.title('Original image')
plt.axis('off')
plt.subplot(132), plt.imshow(fimg2, 'gray'), plt.title('Image spectrum')
plt.axis('off')
# plt.show()

#######high with subtracting

low2=fshift2*guassianLP(10,img2.shape)
lowPass=np.fft.ifftshift(low2)
inverse_low=np.fft.ifft2(lowPass)

high=img2-inverse_low

#### high with filter
HighPassCenter=fshift2*guassianHP(90,img2.shape)
HighPass=np.fft.ifftshift(HighPassCenter)
inverse_Highpass=np.fft.ifft2(HighPass)

plt.subplot(133), plt.imshow(np.abs(inverse_Highpass), 'gray'), plt.title('Guassian HighPass')
plt.axis('off')
# plt.show()
plt.figure()
plt.imshow(np.abs(high),'gray')
plt.title("Guassian HighPass")
plt.show()

plt.imshow(np.log(1+np.abs(HighPassCenter)),cmap='gray')
plt.title("amplitude of FT for filtered image")
plt.show()

hybrid_image=HighPassCenter+LowPassCenter

hybrid=np.fft.ifftshift(hybrid_image)
inverse_hybrid=np.fft.ifft2(hybrid)

plt.imshow(np.abs(inverse_hybrid),cmap='gray')
plt.title("merged image")
plt.show()

plt.imshow(np.log(1+np.abs(hybrid_image)),cmap='gray')
plt.title("FT of merged image")
plt.show()