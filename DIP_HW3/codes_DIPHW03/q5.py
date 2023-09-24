import numpy as np
from scipy import misc
from scipy import fftpack
import matplotlib.pyplot as plt
import cv2 as cv
import math
from skimage.metrics import structural_similarity
# import numpy as np
from scipy.fftpack import fftn, ifftn


img = cv.imread('donald_in_car_2.png',0)
print(img.shape)


kernel = cv.imread('kernel_1.png',0)
print(kernel.shape)
plt.imshow(kernel,cmap='gray')
# plt.figure()
plt.imshow(cv.cvtColor(img,cv.COLOR_BGR2RGB))

# plt.show()
#
sz = (img.shape[0] - kernel.shape[0], img.shape[1] - kernel.shape[1])  # total amount of padding
kernel = np.pad(kernel, (((sz[0]+1)//2, sz[0]//2), ((sz[1]+1)//2, sz[1]//2)),  'constant')
kernel = fftpack.ifftshift(kernel)

filtered=np.real(fftpack.ifft2(fftpack.fft2(img)*fftpack.fft2(kernel)))

plt.imshow(filtered,cmap='gray')
plt.title("convolved image")
plt.show()
#
conjfft_kernel=np.conjugate(fftpack.fft2(kernel))
filtered_fft=fftpack.fft2(filtered)

kernel_fft=fftpack.fft2(kernel)

soorat=np.multiply(conjfft_kernel,filtered_fft)

makhraj=np.multiply(conjfft_kernel,kernel_fft)

kasr=np.divide(soorat,makhraj)

b=np.real(fftpack.ifft2(kasr))

plt.imshow(b,cmap='gray')
plt.title("recovered image by Eq.4")
# plt.show()

############part c
A=np.identity(405, dtype = float)

sz = (img.shape[0] - A.shape[0], img.shape[1] - A.shape[1])  # total amount of padding
A = np.pad(A, (((sz[0]+1)//2, sz[0]//2), ((sz[1]+1)//2, sz[1]//2)), 'constant')

conjfft_A=np.conjugate(fftpack.fft2(A))
A_fft=fftpack.fft2(A)
regul_term=np.multiply(conjfft_A,A_fft)*0.5

makhraj=makhraj+regul_term

kasr=np.divide(soorat,makhraj)

c=np.real(fftpack.ifft2(kasr))

plt.imshow(c,cmap='gray')
plt.title("recovered image by Eq.6")
# plt.show()

def psnr(img1, img2):
    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        return 100
    PIXEL_MAX = 255.0
    return 20 * math.log10(PIXEL_MAX / math.sqrt(mse))


d = psnr(img, c)
print("psnr",d)
(score, diff) = structural_similarity(img, c, full=True , multichannel=True)
diff = (diff*255 ).astype("uint8")
# cv.namedWindow("diff", cv.WINDOW_NORMAL)
# cv.imshow("diff", diff)
print("SSIM:{}".format(score))
# cv.waitKey(0)

############part d

A=cv.Laplacian(img,cv.CV_64F)
print(A.shape)

conjfft_A=np.conjugate(fftpack.fft2(A))
A_fft=fftpack.fft2(A)
regul_term=np.multiply(conjfft_A,A_fft)*0.5

makhraj=makhraj+regul_term

kasr=np.divide(soorat,makhraj)

d=np.real(fftpack.ifft2(kasr))

plt.imshow(d,cmap='gray')
plt.title("recovered image by A=D")
# plt.show()

def psnr(img1, img2):
    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        return 100
    PIXEL_MAX = 255.0
    return 20 * math.log10(PIXEL_MAX / math.sqrt(mse))


psnr_ = psnr(img, d)
print("psnr",psnr_)
(score, diff) = structural_similarity(img, d, full=True , multichannel=True)
diff = (diff*255 ).astype("uint8")
# cv.namedWindow("diff", cv.WINDOW_NORMAL)
# cv.imshow("diff", diff)
print("SSIM:{}".format(score))
# cv.waitKey(0)


############## colored image

img = cv.imread('donald_in_car_2.png')
print(img.shape)


kernel = cv.imread('kernel_1.png',0)
print(kernel.shape)
plt.imshow(kernel,cmap='gray')
# plt.figure()
plt.imshow(cv.cvtColor(img,cv.COLOR_BGR2RGB))

# plt.show()

def psf2otf(psf, otf_size):
    # calculate otf from psf with size >= psf size

    if psf.any():  # if any psf element is non-zero
        # pad PSF with zeros up to image size
        pad_size = ((0, otf_size[0] - psf.shape[0]), (0, otf_size[1] - psf.shape[1]))
        psf_padded = np.pad(psf, pad_size, 'constant')

        # circularly shift psf
        psf_padded = np.roll(psf_padded, -int(np.floor(psf.shape[0] / 2)), axis=0)
        psf_padded = np.roll(psf_padded, -int(np.floor(psf.shape[1] / 2)), axis=1)

        # calculate otf
        otf = np.fft.fftn(psf_padded)
        # this condition depends on psf size
        num_small = np.log2(psf.shape[0]) * 4 * np.spacing(1)
        if np.max(abs(otf.imag)) / np.max(abs(otf)) <= num_small:
            otf = otf.real
    else:  # if all psf elements are zero
        otf = np.zeros(otf_size)
    return otf

otf_kernel=psf2otf(kernel,(801,1200))
# otf_kernel=np.conjugate(otf_kernel)
img1=np.fft.fft2(img[:,:,0])
inner1=np.multiply(otf_kernel,img1)

img2=np.fft.fft2(img[:,:,1])
inner2=np.multiply(otf_kernel,img2)

img3=np.fft.fft2(img[:,:,2])
inner3=np.multiply(otf_kernel,img3)

inner1=np.abs(np.fft.ifft2(inner1))
inner2=np.abs(np.fft.ifft2(inner2))
inner3=np.abs(np.fft.ifft2(inner3))


inner1=((inner1 - inner1.min()) /(inner1.max() - inner1.min()))
inner2=((inner2 - inner2.min()) /(inner2.max() - inner2.min()))
inner3=((inner3 - inner3.min()) /(inner3.max() - inner3.min()))


convoloved=np.empty((img.shape[0],img.shape[1],3))
convoloved[:,:,0]=inner3
convoloved[:,:,1]=inner2
convoloved[:,:,2]=inner1


plt.imshow(convoloved)
plt.title("convolved image")
# plt.show()



otf_kernel=np.matrix(psf2otf(kernel,(801,1200)))
otf_kernel=np.conjugate(otf_kernel)
img1=np.fft.fft2(convoloved[:,:,0])

ch1=np.multiply(otf_kernel,img1)/np.multiply(otf_kernel,psf2otf(kernel,(801,1200)))

img2=np.fft.fft2(convoloved[:,:,1])
ch2=np.multiply(otf_kernel,img2)/np.multiply(otf_kernel,psf2otf(kernel,(801,1200)))

img3=np.fft.fft2(convoloved[:,:,2])
ch3=np.multiply(otf_kernel,img3)/np.multiply(otf_kernel,psf2otf(kernel,(801,1200)))

inner1=np.abs(np.fft.ifft2(ch1))
inner2=np.abs(np.fft.ifft2(ch2))
inner3=np.abs(np.fft.ifft2(ch3))

# inner1=np.abs(inner1)
# inner2=np.abs(inner2)
# inner3=np.abs(inner3)

recover1=np.empty((img.shape[0],img.shape[1],3))

inner1=((inner1 - inner1.min()) /(inner1.max() - inner1.min()))
inner2=((inner2 - inner2.min()) /(inner2.max() - inner2.min()))
inner3=((inner3 - inner3.min()) /(inner3.max() - inner3.min()))

recover1[:,:,0]=inner1
recover1[:,:,1]=inner2
recover1[:,:,2]=inner3

plt.imshow(recover1)
plt.title("recovered by Eq.4")
# plt.show()

#####Eq 6

A=np.identity(801, dtype = float)

sz = (img.shape[0] - A.shape[0], img.shape[1] - A.shape[1])  # total amount of padding
A = np.pad(A, (((sz[0]+1)//2, sz[0]//2), ((sz[1]+1)//2, sz[1]//2)), 'constant')

conjfft_A=np.conjugate(fftpack.fft2(A))
A_fft=fftpack.fft2(A)
regul_term=np.multiply(conjfft_A,A_fft)*0.5


otf_kernel=np.matrix(psf2otf(kernel,(801,1200)))
otf_kernel=np.conjugate(otf_kernel)
img1=np.fft.fft2(convoloved[:,:,0])

ch1=np.multiply(otf_kernel,img1)/(np.multiply(otf_kernel,psf2otf(kernel,(801,1200)))+regul_term)

img2=np.fft.fft2(convoloved[:,:,1])
ch2=np.multiply(otf_kernel,img2)/(np.multiply(otf_kernel,psf2otf(kernel,(801,1200)))+regul_term)

img3=np.fft.fft2(convoloved[:,:,2])
ch3=np.multiply(otf_kernel,img3)/(np.multiply(otf_kernel,psf2otf(kernel,(801,1200)))+regul_term)

inner1=np.abs(np.fft.ifft2(ch1))
inner2=np.abs(np.fft.ifft2(ch2))
inner3=np.abs(np.fft.ifft2(ch3))

recover2=np.empty((img.shape[0],img.shape[1],3))

inner1=((inner1 - inner1.min()) /(inner1.max() - inner1.min()))
inner2=((inner2 - inner2.min()) /(inner2.max() - inner2.min()))
inner3=((inner3 - inner3.min()) /(inner3.max() - inner3.min()))

recover2[:,:,0]=inner1
recover2[:,:,1]=inner2
recover2[:,:,2]=inner3

plt.imshow(recover2)
plt.title("recovered by Eq.6")
# plt.show()

psnr_ = psnr(img, recover2)
print("psnr",psnr_)
(score, diff) = structural_similarity(img, recover2, full=True , multichannel=True)
diff = (diff*255 ).astype("uint8")
# cv.namedWindow("diff", cv.WINDOW_NORMAL)
# cv.imshow("diff", diff)
print("SSIM:{}".format(score))
# cv.waitKey(0)

####Eq6 A=D
img_gray = cv.imread('donald_in_car_2.png',0)
A=cv.Laplacian(img_gray,cv.CV_64F)
print(A.shape)

conjfft_A=np.conjugate(fftpack.fft2(A))
A_fft=fftpack.fft2(A)
regul_term=np.multiply(conjfft_A,A_fft)*0.5


otf_kernel=np.matrix(psf2otf(kernel,(801,1200)))
otf_kernel=np.conjugate(otf_kernel)
img1=np.fft.fft2(convoloved[:,:,0])

ch1=np.multiply(otf_kernel,img1)/(np.multiply(otf_kernel,psf2otf(kernel,(801,1200)))+regul_term)

img2=np.fft.fft2(convoloved[:,:,1])
ch2=np.multiply(otf_kernel,img2)/(np.multiply(otf_kernel,psf2otf(kernel,(801,1200)))+regul_term)

img3=np.fft.fft2(convoloved[:,:,2])
ch3=np.multiply(otf_kernel,img3)/(np.multiply(otf_kernel,psf2otf(kernel,(801,1200)))+regul_term)

inner1=np.abs(np.fft.ifft2(ch1))
inner2=np.abs(np.fft.ifft2(ch2))
inner3=np.abs(np.fft.ifft2(ch3))

recover3=np.empty((img.shape[0],img.shape[1],3))

inner1=((inner1 - inner1.min()) /(inner1.max() - inner1.min()))
inner2=((inner2 - inner2.min()) /(inner2.max() - inner2.min()))
inner3=((inner3 - inner3.min()) /(inner3.max() - inner3.min()))

recover3[:,:,0]=inner1
recover3[:,:,1]=inner2
recover3[:,:,2]=inner3

plt.imshow(recover3)
plt.title("recovered by Eq.6 : A=D")
# plt.show()

psnr_ = psnr(img, recover3)
print("psnr",psnr_)
(score, diff) = structural_similarity(img, recover3, full=True , multichannel=True)
diff = (diff*255 ).astype("uint8")
# cv.namedWindow("diff", cv.WINDOW_NORMAL)
# cv.imshow("diff", diff)
print("SSIM:{}".format(score))
# cv.waitKey(0)
######################## part e

# convoloved=cv.cvtColor(convoloved,cv.COLOR_RGB2GRAY)
f1 = np.fft.fft2(filtered)
fshift_filtered = np.fft.fftshift(f1)
# fft result is a complex number, its absolute value result is amplitude
fimg1 = np.log(np.abs(fshift_filtered))

plt.subplot(121), plt.imshow(filtered,cmap='gray'), plt.title('Original image')
plt.axis('off')
plt.subplot(122), plt.imshow(fimg1, 'gray'), plt.title('Image spectrum')
plt.axis('off')
plt.show()

img = cv.imread('donald_in_car_2.png',0)
print(img.shape)

f1 = np.fft.fft2(img)
fshift_origin = np.fft.fftshift(f1)
# fft result is a complex number, its absolute value result is amplitude
fimg1 = np.log(np.abs(fshift_origin))

plt.subplot(121), plt.imshow(img,cmap='gray'), plt.title('Original image')
plt.axis('off')
plt.subplot(122), plt.imshow(fimg1, 'gray'), plt.title('Image spectrum')
plt.axis('off')
plt.show()

fft_kernel=np.divide(fshift_filtered,fshift_origin)

plt.imshow(np.log(np.abs(fft_kernel)),cmap='gray')
# plt.imshow()
plt.show()
# otf_kernel = np.real(np.fft.ifft2(fft_kernel))


def otf2psf(otf, psf_size):
    # calculate psf from otf with size <= otf size

    if otf.any():  # if any otf element is non-zero
        # calculate psf
        psf = ifftn(otf)
        # this condition depends on psf size
        num_small = np.log2(otf.shape[0]) * 4 * np.spacing(1)
        if np.max(abs(psf.imag)) / np.max(abs(psf)) <= num_small:
            psf = psf.real

            # circularly shift psf
        psf = np.roll(psf, int(np.floor(psf_size[0] / 2)), axis=0)
        psf = np.roll(psf, int(np.floor(psf_size[1] / 2)), axis=1)

        # crop psf
        psf = psf[0:psf_size[0], 0:psf_size[1]]
    else:  # if all otf elements are zero
        psf = np.zeros(psf_size)
    return psf

kernel = cv.imread('kernel_1.png',0)
psf_kernel=otf2psf(fft_kernel,(kernel.shape[0],kernel.shape[1]))

# psf_kernel = np.real(np.fft.ifft2(psf_kernel))
plt.imshow(np.abs(psf_kernel),cmap='gray')
plt.show()


# print()
plt.subplot(121), plt.imshow(np.abs(psf_kernel),cmap='gray'), plt.title('found kernel')
plt.axis('off')
plt.subplot(122), plt.imshow(kernel, 'gray'), plt.title('original kernel')
plt.axis('off')
plt.show()
print("RMSE:",np.abs(np.sqrt(np.mean((np.abs(psf_kernel)-kernel)**2))))

########### by Eq.4

conjfft_kernel=np.conjugate(fftpack.fft2(img))
filtered_fft=fftpack.fft2(filtered)

kernel_fft=fftpack.fft2(img)

soorat=np.multiply(conjfft_kernel,filtered_fft)

makhraj=np.multiply(conjfft_kernel,kernel_fft)

kasr=np.divide(soorat,makhraj)

b=np.abs(fftpack.ifftshift(fftpack.ifft2(kasr)))

result_kernel=np.empty((kernel.shape))
nb = b.shape[0]
na = b.shape[1]
lower1 = (nb // 2) - (kernel.shape[0] // 2)
upper1 = (nb // 2) + (kernel.shape[0] // 2)

lower2 = (na // 2) - (kernel.shape[1] // 2)
upper2 = (na // 2) + (kernel.shape[1] // 2)
result_kernel[:,:]= b[lower1+1:upper1+2, lower2:upper2+1]


# plt.imshow(b,cmap='gray')
#
# plt.title("recovered mask by Eq.4")
# plt.show()
#
# plt.imshow(result_kernel,cmap='gray')
#
# plt.title("recovered mask by Eq.4")
# plt.show()


plt.subplot(121), plt.imshow(result_kernel,cmap='gray'), plt.title('found kernel by Eq.4')
plt.axis('off')
plt.subplot(122), plt.imshow(kernel, 'gray'), plt.title('original kernel')
plt.axis('off')
plt.show()

print("RMSE:",np.abs(np.sqrt(np.mean((np.abs(result_kernel)-kernel)**2))))





