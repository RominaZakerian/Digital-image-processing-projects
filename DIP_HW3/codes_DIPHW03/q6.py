from scipy.ndimage import gaussian_filter
from scipy import misc
import matplotlib.pyplot as plt
import cv2 as cv
import numpy as np
from scipy import fftpack
import skimage

A = cv.imread('audi_q7.png')
print(A.shape)
A = cv.cvtColor(A, cv.COLOR_BGR2GRAY)
print(A.shape)
A = skimage.img_as_float(A)

kernel = cv.imread('audi_saipa_mask.png')
# plt.imshow(kernel,cmap='gray')
# plt.figure()
kernel = cv.cvtColor(kernel, cv.COLOR_BGR2GRAY)
kernel = skimage.img_as_float(kernel)

B = cv.imread('saipa_151.png')
print(B.shape)
B= cv.cvtColor(B, cv.COLOR_BGR2GRAY)
B = skimage.img_as_float(B)
#########image A

plt.gray()  # show the filtered result in grayscale

def gaussian_LP(size,sigma):
    center = np.array([size[0]/2 , size[1]/2])
    Xs,Ys=np.meshgrid(np.arange(size[1]) , np.arange(size[0]))
    indice_map=np.dstack([Ys,Xs])
    lpf=np.exp(-np.power(indice_map-center , 2).sum(axis=2)/(2*np.power(sigma,2)))
    return lpf

fig = plt.figure()
fig.subplots_adjust(hspace=0.4, wspace=0.4)
LAP_A=np.empty((6,A.shape[0],A.shape[1]))
# A=fftpack.ifftshift(A)
for i in range(6):
    ax = fig.add_subplot(2, 3, i+1)
    g1 = gaussian_LP(A.shape, 2 ** (i))
    A_fft = np.fft.fft2(A)
    A_fft_shift = np.fft.fftshift(A_fft)
    filtered_img1 = A_fft_shift * g1
    Lap1 = np.abs(np.fft.ifft2(np.fft.ifftshift(filtered_img1)))

    g2 = gaussian_LP(A.shape, 2 ** (i + 1))
    A_fft = np.fft.fft2(A)
    A_fft_shift = np.fft.fftshift(A_fft)
    filtered_img1 = A_fft_shift * g2
    Lap2 = np.abs(np.fft.ifft2(np.fft.ifftshift(filtered_img1)))
    sub = Lap2 - Lap1
    LAP_A[i,:,:]=sub
    ax.imshow(sub)
    # plt.imshow(sub)
    # ax.title("lap" + str(i+1))
    plt.title("laplacian" + str(i + 1))
# plt.title("Laplacian of image in different levels")
plt.show()

##########laplacian B
fig = plt.figure()
fig.subplots_adjust(hspace=0.4, wspace=0.4)
LAP_B=np.empty((6,B.shape[0],B.shape[1]))
for i in range(6):
    ax = fig.add_subplot(2, 3, i + 1)
    g1 = gaussian_LP(B.shape, 2 ** (i))
    B_fft = np.fft.fft2(B)
    B_fft_shift = np.fft.fftshift(B_fft)
    filtered_img1 = B_fft_shift * g1
    Lap1 = np.abs(np.fft.ifft2(np.fft.ifftshift(filtered_img1)))

    g2 = gaussian_LP(B.shape, 2 ** (i + 1))
    B_fft = np.fft.fft2(B)
    B_fft_shift = np.fft.fftshift(B_fft)
    filtered_img1 = B_fft_shift * g2
    Lap2 = np.abs(np.fft.ifft2(np.fft.ifftshift(filtered_img1)))
    sub = Lap2 - Lap1
    LAP_B[i,:,:]=sub
    # plt.imshow(sub)
    # plt.title("lap" + str(i + 1))
    # plt.show()
    ax.imshow(sub)
    # plt.imshow(sub)
    # ax.title("lap" + str(i+1))
    plt.title("laplacian" + str(i + 1))

# plt.title("Laplacian of image in different levels")
plt.show()


########guassian of mask
fig = plt.figure()
fig.subplots_adjust(hspace=0.4, wspace=0.4)
guassi_mask=np.empty((6,kernel.shape[0],kernel.shape[1]))
print(kernel.shape)
for i in range(6):
    ax = fig.add_subplot(2, 3, i + 1)
    g1 = gaussian_LP(A.shape, 2 ** (i))
    A_fft = np.fft.fft2(kernel)
    A_fft_shift = np.fft.fftshift(A_fft)
    filtered_img1 = A_fft_shift * g1
    kernel_conv = np.abs(np.fft.ifft2(np.fft.ifftshift(filtered_img1)))
    guassi_mask[i,:,:]=kernel_conv
    # plt.imshow(kernel_conv)
    # plt.title("kernel" + str(i + 1))
    # plt.show()
    ax.imshow(kernel_conv)
    # plt.imshow(sub)
    plt.title("gaussian" + str(i + 1))

# plt.title("gaussian of mask in different levels")
# plt.show()

######## combining
combined_image=np.empty((6,256,576))
summation=np.zeros((256,576))

c = (guassi_mask*LAP_A+(1-guassi_mask)*LAP_B).sum(axis=0)
print(c.shape)

c=((c - c.min()) /(c.max() - c.min()))


# result=np.sum(combined_image,axis=0)
# plt.imshow(c,cmap='gray')
# plt.show()


##############  MESSI WHO?


A = cv.imread('messi.png')
print("A",A.shape)
# A = cv.cvtColor(A, cv.COLOR_BGR2GRAY)
# print("A",A.shape)
A = skimage.img_as_float(A)
A_ch1=A[:,:,0]
A_ch2=A[:,:,1]
A_ch3=A[:,:,2]
# plt.imshow(A)
# plt.figure()

kernel = cv.imread('messi_trump_mask.png')
print("kernel",kernel.shape)
# plt.imshow(kernel,cmap='gray')
# plt.figure()
# kernel = cv.cvtColor(kernel, cv.COLOR_BGR2GRAY)
kernel = skimage.img_as_float(kernel)
kernel_ch1=kernel[:,:,0]
kernel_ch2=kernel[:,:,1]
kernel_ch3=kernel[:,:,2]

B = cv.imread('trump.png')
print("B",B.shape)
# plt.imshow(B)
# plt.show()
# B= cv.cvtColor(B, cv.COLOR_BGR2GRAY)
B = skimage.img_as_float(B)
B_ch1=B[:,:,0]
B_ch2=B[:,:,1]
B_ch3=B[:,:,2]
#########image A
#
# plt.gray()  # show the filtered result in grayscale
#
def gaussian_LP(size,sigma):
    center = np.array([size[0]/2 , size[1]/2])
    Xs,Ys=np.meshgrid(np.arange(size[1]) , np.arange(size[0]))
    indice_map=np.dstack([Ys,Xs])
    lpf=np.exp(-np.power(indice_map-center , 2).sum(axis=2)/(2*np.power(sigma,2)))
    return lpf

fig = plt.figure()
fig.subplots_adjust(hspace=0.4, wspace=0.4)
LAP_A_ch1=np.empty((6,A_ch1.shape[0],A_ch1.shape[1]))
# A=fftpack.ifftshift(A)
for i in range(6):
    ax = fig.add_subplot(2, 3, i + 1)
    g1 = gaussian_LP(A_ch1.shape, 2 ** (i))
    A_ch1_fft = np.fft.fft2(A_ch1)
    A_ch1_fft_shift = np.fft.fftshift(A_ch1_fft)
    filtered_img1 = A_ch1_fft_shift * g1
    Lap1 = np.abs(np.fft.ifft2(np.fft.ifftshift(filtered_img1)))

    g2 = gaussian_LP(A_ch1.shape, 2 ** (i + 1))
    A_ch1_fft = np.fft.fft2(A_ch1)
    A_ch1_fft_shift = np.fft.fftshift(A_ch1_fft)
    filtered_img1 = A_ch1_fft_shift * g2
    Lap2 = np.abs(np.fft.ifft2(np.fft.ifftshift(filtered_img1)))
    sub = Lap2 - Lap1
    LAP_A_ch1[i,:,:]=sub
    # plt.imshow(sub)
    # plt.title("lap" + str(i+1))
    # plt.show()
    ax.imshow(sub)
    # plt.imshow(sub)
    plt.title("Laplacian " + str(i + 1) + " of channel_1")

# plt.title("gaussian of mask in different levels")
plt.show()

fig = plt.figure()
fig.subplots_adjust(hspace=0.4, wspace=0.4)
LAP_A_ch2=np.empty((6,A_ch2.shape[0],A_ch2.shape[1]))
# A=fftpack.ifftshift(A)
for i in range(6):
    ax = fig.add_subplot(2, 3, i + 1)
    g1 = gaussian_LP(A_ch2.shape, 2 ** (i))
    A_ch2_fft = np.fft.fft2(A_ch2)
    A_ch2_fft_shift = np.fft.fftshift(A_ch2_fft)
    filtered_img1 = A_ch2_fft_shift * g1
    Lap1 = np.abs(np.fft.ifft2(np.fft.ifftshift(filtered_img1)))

    g2 = gaussian_LP(A_ch2.shape, 2 ** (i + 1))
    A_ch2_fft = np.fft.fft2(A_ch2)
    A_ch2_fft_shift = np.fft.fftshift(A_ch2_fft)
    filtered_img1 = A_ch2_fft_shift * g2
    Lap2 = np.abs(np.fft.ifft2(np.fft.ifftshift(filtered_img1)))
    sub = Lap2 - Lap1
    LAP_A_ch2[i,:,:]=sub
    # plt.imshow(sub)
    # plt.title("lap" + str(i+1))
    # plt.show()
    ax.imshow(sub)
    # plt.imshow(sub)
    plt.title("Laplacian " + str(i + 1)+ " of channel_2")

# plt.title("gaussian of mask in different levels")
plt.show()

fig = plt.figure()
fig.subplots_adjust(hspace=0.4, wspace=0.4)
LAP_A_ch3=np.empty((6,A_ch3.shape[0],A_ch3.shape[1]))
# A=fftpack.ifftshift(A)
for i in range(6):
    ax = fig.add_subplot(2, 3, i + 1)
    g1 = gaussian_LP(A_ch3.shape, 2 ** (i))
    A_ch3_fft = np.fft.fft2(A_ch3)
    A_ch3_fft_shift = np.fft.fftshift(A_ch3_fft)
    filtered_img1 = A_ch3_fft_shift * g1
    Lap1 = np.abs(np.fft.ifft2(np.fft.ifftshift(filtered_img1)))

    g2 = gaussian_LP(A_ch3.shape, 2 ** (i + 1))
    A_ch3_fft = np.fft.fft2(A_ch3)
    A_ch3_fft_shift = np.fft.fftshift(A_ch3_fft)
    filtered_img1 = A_ch3_fft_shift * g2
    Lap2 = np.abs(np.fft.ifft2(np.fft.ifftshift(filtered_img1)))
    sub = Lap2 - Lap1
    LAP_A_ch3[i,:,:]=sub
    # plt.imshow(sub)
    # plt.title("lap" + str(i+1))
    # plt.show()
    ax.imshow(sub)
    # plt.imshow(sub)
    plt.title("Laplacian " + str(i + 1)+" of channel_3")

# plt.title("gaussian of mask in different levels")
plt.show()

# ##########laplacian B
fig = plt.figure()
fig.subplots_adjust(hspace=0.4, wspace=0.4)
LAP_B_ch1=np.empty((6,B_ch1.shape[0],B_ch1.shape[1]))
# A=fftpack.ifftshift(A)
for i in range(6):
    ax = fig.add_subplot(2, 3, i + 1)
    g1 = gaussian_LP(B_ch1.shape, 2 ** (i))
    B_ch1_fft = np.fft.fft2(B_ch1)
    B_ch1_fft_shift = np.fft.fftshift(B_ch1_fft)
    filtered_img1 = B_ch1_fft_shift * g1
    Lap1 = np.abs(np.fft.ifft2(np.fft.ifftshift(filtered_img1)))

    g2 = gaussian_LP(B_ch1.shape, 2 ** (i + 1))
    B_ch1_fft = np.fft.fft2(B_ch1)
    B_ch1_fft_shift = np.fft.fftshift(B_ch1_fft)
    filtered_img1 = B_ch1_fft_shift * g2
    Lap2 = np.abs(np.fft.ifft2(np.fft.ifftshift(filtered_img1)))
    sub = Lap2 - Lap1
    LAP_B_ch1[i,:,:]=sub
    # plt.imshow(sub)
    # plt.title("lap" + str(i+1))
    # plt.show()
    ax.imshow(sub)
    # plt.imshow(sub)
    plt.title("Laplacian " + str(i + 1) + " of channel_1")

# plt.title("gaussian of mask in different levels")
plt.show()


fig = plt.figure()
fig.subplots_adjust(hspace=0.4, wspace=0.4)
LAP_B_ch2=np.empty((6,B_ch2.shape[0],B_ch2.shape[1]))
# A=fftpack.ifftshift(A)
for i in range(6):
    ax = fig.add_subplot(2, 3, i + 1)
    g1 = gaussian_LP(B_ch2.shape, 2 ** (i))
    B_ch2_fft = np.fft.fft2(B_ch2)
    B_ch2_fft_shift = np.fft.fftshift(B_ch2_fft)
    filtered_img1 = B_ch2_fft_shift * g1
    Lap1 = np.abs(np.fft.ifft2(np.fft.ifftshift(filtered_img1)))

    g2 = gaussian_LP(B_ch2.shape, 2 ** (i + 1))
    B_ch2_fft = np.fft.fft2(B_ch2)
    B_ch2_fft_shift = np.fft.fftshift(B_ch2_fft)
    filtered_img1 = B_ch2_fft_shift * g2
    Lap2 = np.abs(np.fft.ifft2(np.fft.ifftshift(filtered_img1)))
    sub = Lap2 - Lap1
    LAP_B_ch2[i,:,:]=sub
    # plt.imshow(sub)
    # plt.title("lap" + str(i+1))
    # plt.show()
    ax.imshow(sub)
    # plt.imshow(sub)
    plt.title("Laplacian " + str(i + 1) +" of channel_2")

# plt.title("gaussian of mask in different levels")
plt.show()


fig = plt.figure()
fig.subplots_adjust(hspace=0.4, wspace=0.4)
LAP_B_ch3=np.empty((6,B_ch3.shape[0],B_ch3.shape[1]))
for i in range(6):
    ax = fig.add_subplot(2, 3, i + 1)
    g1 = gaussian_LP(B_ch3.shape, 2 ** (i))
    B_ch3_fft = np.fft.fft2(B_ch3)
    B_ch3_fft_shift = np.fft.fftshift(B_ch3_fft)
    filtered_img1 = B_ch3_fft_shift * g1
    Lap1 = np.abs(np.fft.ifft2(np.fft.ifftshift(filtered_img1)))

    g2 = gaussian_LP(B_ch3.shape, 2 ** (i + 1))
    B_ch3_fft = np.fft.fft2(B_ch3)
    B_ch3_fft_shift = np.fft.fftshift(B_ch3_fft)
    filtered_img1 = B_ch3_fft_shift * g2
    Lap2 = np.abs(np.fft.ifft2(np.fft.ifftshift(filtered_img1)))
    sub = Lap2 - Lap1
    LAP_B_ch3[i,:,:]=sub
    # plt.imshow(sub)
    # plt.title("lap" + str(i+1))
    # plt.show()
    ax.imshow(sub)
    # plt.imshow(sub)
    plt.title("Laplacian " + str(i + 1) + " of channel_3")

# plt.title("gaussian of mask in different levels")
plt.show()

# ########guassian of mask
fig = plt.figure()
fig.subplots_adjust(hspace=0.4, wspace=0.4)
guassi_mask_ch1=np.empty((6,kernel_ch1.shape[0],kernel_ch1.shape[1]))
print(kernel_ch1.shape)
for i in range(6):
    ax = fig.add_subplot(2, 3, i + 1)
    g1 = gaussian_LP(kernel_ch1.shape, 2 ** (i))
    A_fft = np.fft.fft2(kernel_ch1)
    A_fft_shift = np.fft.fftshift(A_fft)
    filtered_img1 = A_fft_shift * g1
    kernel_conv = np.abs(np.fft.ifft2(np.fft.ifftshift(filtered_img1)))
    guassi_mask_ch1[i,:,:]=kernel_conv
    # plt.imshow(kernel_conv)
    # plt.title("kernel" + str(i + 1))
    # plt.show()
    ax.imshow(kernel_conv)
    # plt.imshow(sub)
    plt.title("gaussian of mask in level " + str(i + 1) + " channel_1")

# plt.title("gaussian of mask in different levels")
plt.show()

fig = plt.figure()
fig.subplots_adjust(hspace=0.4, wspace=0.4)
guassi_mask_ch2=np.empty((6,kernel_ch2.shape[0],kernel_ch2.shape[1]))
print(kernel_ch2.shape)
for i in range(6):
    ax = fig.add_subplot(2, 3, i + 1)
    g1 = gaussian_LP(kernel_ch2.shape, 2 ** (i))
    A_fft = np.fft.fft2(kernel_ch2)
    A_fft_shift = np.fft.fftshift(A_fft)
    filtered_img1 = A_fft_shift * g1
    kernel_conv = np.abs(np.fft.ifft2(np.fft.ifftshift(filtered_img1)))
    guassi_mask_ch2[i,:,:]=kernel_conv
    # plt.imshow(kernel_conv)
    # plt.title("kernel" + str(i + 1))
    # plt.show()
    ax.imshow(kernel_conv)
    # plt.imshow(sub)
    plt.title("gaussian of mask in level " + str(i + 1) + " channel_2")

# plt.title("gaussian of mask in different levels")
plt.show()

fig = plt.figure()
fig.subplots_adjust(hspace=0.4, wspace=0.4)
guassi_mask_ch3=np.empty((6,kernel_ch3.shape[0],kernel_ch3.shape[1]))
print(kernel_ch3.shape)
for i in range(6):
    ax = fig.add_subplot(2, 3, i + 1)
    g1 = gaussian_LP(kernel_ch3.shape, 2 ** (i))
    A_fft = np.fft.fft2(kernel_ch3)
    A_fft_shift = np.fft.fftshift(A_fft)
    filtered_img1 = A_fft_shift * g1
    kernel_conv = np.abs(np.fft.ifft2(np.fft.ifftshift(filtered_img1)))
    guassi_mask_ch3[i,:,:]=kernel_conv
    # plt.imshow(kernel_conv)
    # plt.title("kernel" + str(i + 1))
    # plt.show()
    ax.imshow(kernel_conv)
    # plt.imshow(sub)
    plt.title("gaussian of mask in level" + str(i + 1)+" channel3")

# plt.title("gaussian of mask in different levels")
plt.show()
# ######## combining
combined_image_2=np.empty((500,620,3))
# summation_1=np.zeros((500,620))

combine_ch1 = (guassi_mask_ch1*LAP_B_ch1+(1-guassi_mask_ch1)*LAP_A_ch1).sum(axis=0)
print(combine_ch1.shape)

combine_ch1=((combine_ch1 - combine_ch1.min()) /(combine_ch1.max() - combine_ch1.min()))

combine_ch2 = (guassi_mask_ch2*LAP_B_ch2+(1-guassi_mask_ch2)*LAP_A_ch2).sum(axis=0)
print(combine_ch2.shape)

combine_ch2=((combine_ch2 - combine_ch2.min()) /(combine_ch2.max() - combine_ch2.min()))

combine_ch3 = (guassi_mask_ch3*LAP_B_ch3+(1-guassi_mask_ch3)*LAP_A_ch3).sum(axis=0)
print(combine_ch3.shape)

combine_ch3=((combine_ch3 - combine_ch3.min()) /(combine_ch3.max() - combine_ch3.min()))

combined_image_2[:,:,0]=combine_ch3
combined_image_2[:,:,1]=combine_ch2
combined_image_2[:,:,2]=combine_ch1

plt.imshow(combined_image_2)
plt.show()

###########T.shirt

A = cv.imread('t-shirt.png')
A= cv.cvtColor(A, cv.COLOR_BGR2RGB)
print("t_shirt",A.shape)

A_ch1=A[:,:,0]
A_ch2=A[:,:,1]
A_ch3=A[:,:,2]
plt.imshow(A)
# plt.figure()
plt.show()
a_file = open("readme.txt", "w")
for row in A:
    np.savetxt(a_file, row)

a_file.close()

B = cv.imread('keep_calm.png')
print("this is B",B)
B= cv.cvtColor(B, cv.COLOR_BGR2RGB)
print("B",B.shape)
plt.imshow(B)
plt.show()


kernel=np.zeros((A.shape))
print("kernel",kernel.shape)

nb = kernel.shape[0]
na = B.shape[0]

nb2 = kernel.shape[1]
na2 = B.shape[1]
print(nb,na)
lower1 = (nb // 2) - (na // 2)
upper1 = (nb // 2) + (na // 2)

lower2 = (nb2 // 2) - (na2 // 2)
upper2 = (nb2 // 2) + (na2 // 2)
kernel[lower1:upper1, lower2:upper2,0] = B[:,:,0]
kernel[lower1:upper1, lower2:upper2,1] = B[:,:,1]
kernel[lower1:upper1, lower2:upper2,2] = B[:,:,2]

plt.imshow(kernel.astype(np.uint8))
plt.title("after editing")
plt.show()
B_created=np.copy(kernel)
# B_created = skimage.img_as_int(B_created)
print(B_created.shape)
# print(B_created[:,:,0])

h = (A.shape[0]-B.shape[0])//2
w = (A.shape[1]-B.shape[1])//2

mask=np.empty((A.shape[0],A.shape[1],3))
for i in range(mask.shape[0]):
    for j in range(mask.shape[1]):
        if  h<i<(A.shape[0]-h) and w<j<(A.shape[1]-w):
            mask[i,j,:] = 1
        else:
            mask[i,j,:] = 0

# windows=np.empty((A.shape[0],A.shape[1],3))
# for i in range(windows.shape[0]):
#     for j in range(windows.shape[1]):
#         if  h<i<(A.shape[0]-h) and w<j<(A.shape[1]-w):
#             windows[i,j,:] = kernel[i,j,:]
#         else:
#             windows[i,j,:] = 0
# plt.imshow(windows)
# print(B.shape)
# plt.title("this is B")
# plt.show()
for i in range(B_created.shape[0]):
    for j in range(B_created.shape[1]):
        if (B_created[i,j,:].all())==0:
            B_created[i,j,:]=[167,2,36]
        else:
            B_created[i, j, :] = kernel[i,j,:]


plt.imshow(B_created.astype(np.uint8))
plt.title("B")
plt.show()

B_ch1=B_created[:,:,0]
B_ch2=B_created[:,:,1]
B_ch3=B_created[:,:,2]

for i in range(kernel.shape[0]):
    for j in range(kernel.shape[1]):
        if (kernel[i,j,:].all())!=0:
            kernel[i,j,:]=[255,255,255]

plt.imshow(kernel.astype(np.uint8))
plt.title("after creating mask")
plt.show()


print("kernel",kernel.shape)

# kernel = skimage.img_as_float(kernel)
kernel_ch1=mask[:,:,0]
kernel_ch2=mask[:,:,1]
kernel_ch3=mask[:,:,2]

#########image A

def gaussian_LP(size,sigma):
    center = np.array([size[0]/2 , size[1]/2])
    Xs,Ys=np.meshgrid(np.arange(size[1]) , np.arange(size[0]))
    indice_map=np.dstack([Ys,Xs])
    lpf=np.exp(-np.power(indice_map-center , 2).sum(axis=2)/(2*np.power(sigma,2)))
    return lpf

fig = plt.figure()
fig.subplots_adjust(hspace=0.4, wspace=0.4)
LAP_A_ch1=np.empty((6,A_ch1.shape[0],A_ch1.shape[1]))
# A=fftpack.ifftshift(A)
for i in range(6):
    ax = fig.add_subplot(2, 3, i + 1)
    g1 = gaussian_LP(A_ch1.shape, 2 ** (i))
    A_ch1_fft = np.fft.fft2(A_ch1)
    A_ch1_fft_shift = np.fft.fftshift(A_ch1_fft)
    filtered_img1 = A_ch1_fft_shift * g1
    Lap1 = np.abs(np.fft.ifft2(np.fft.ifftshift(filtered_img1)))

    g2 = gaussian_LP(A_ch1.shape, 2 ** (i + 1))
    A_ch1_fft = np.fft.fft2(A_ch1)
    A_ch1_fft_shift = np.fft.fftshift(A_ch1_fft)
    filtered_img1 = A_ch1_fft_shift * g2
    Lap2 = np.abs(np.fft.ifft2(np.fft.ifftshift(filtered_img1)))
    sub = Lap2 - Lap1
    LAP_A_ch1[i,:,:]=sub
    # plt.imshow(sub)
    # plt.title("lap" + str(i+1))
    # plt.show()
    ax.imshow(sub)
    # plt.imshow(sub)
    plt.title("Laplacian " + str(i + 1) + " channel1")

# plt.title("gaussian of mask in different levels")
plt.show()

fig = plt.figure()
fig.subplots_adjust(hspace=0.4, wspace=0.4)
LAP_A_ch2=np.empty((6,A_ch2.shape[0],A_ch2.shape[1]))
# A=fftpack.ifftshift(A)
for i in range(6):
    ax = fig.add_subplot(2, 3, i + 1)
    g1 = gaussian_LP(A_ch2.shape, 2 ** (i))
    A_ch2_fft = np.fft.fft2(A_ch2)
    A_ch2_fft_shift = np.fft.fftshift(A_ch2_fft)
    filtered_img1 = A_ch2_fft_shift * g1
    Lap1 = np.abs(np.fft.ifft2(np.fft.ifftshift(filtered_img1)))

    g2 = gaussian_LP(A_ch2.shape, 2 ** (i + 1))
    A_ch2_fft = np.fft.fft2(A_ch2)
    A_ch2_fft_shift = np.fft.fftshift(A_ch2_fft)
    filtered_img1 = A_ch2_fft_shift * g2
    Lap2 = np.abs(np.fft.ifft2(np.fft.ifftshift(filtered_img1)))
    sub = Lap2 - Lap1
    LAP_A_ch2[i,:,:]=sub
    # plt.imshow(sub)
    # plt.title("lap" + str(i+1))
    # plt.show()
    ax.imshow(sub)
    # plt.imshow(sub)
    plt.title("Laplacian " + str(i + 1) + " channel2")
plt.show()

fig = plt.figure()
fig.subplots_adjust(hspace=0.4, wspace=0.4)
LAP_A_ch3=np.empty((6,A_ch3.shape[0],A_ch3.shape[1]))
# A=fftpack.ifftshift(A)
for i in range(6):
    ax = fig.add_subplot(2, 3, i + 1)
    g1 = gaussian_LP(A_ch3.shape, 2 ** (i))
    A_ch3_fft = np.fft.fft2(A_ch3)
    A_ch3_fft_shift = np.fft.fftshift(A_ch3_fft)
    filtered_img1 = A_ch3_fft_shift * g1
    Lap1 = np.abs(np.fft.ifft2(np.fft.ifftshift(filtered_img1)))

    g2 = gaussian_LP(A_ch3.shape, 2 ** (i + 1))
    A_ch3_fft = np.fft.fft2(A_ch3)
    A_ch3_fft_shift = np.fft.fftshift(A_ch3_fft)
    filtered_img1 = A_ch3_fft_shift * g2
    Lap2 = np.abs(np.fft.ifft2(np.fft.ifftshift(filtered_img1)))
    sub = Lap2 - Lap1
    LAP_A_ch3[i,:,:]=sub
    # plt.imshow(sub)
    # plt.title("lap" + str(i+1))
    # plt.show()
    ax.imshow(sub)
    # plt.imshow(sub)
    plt.title("Laplacian " + str(i + 1) + " channel3")
plt.show()

# ##########laplacian B

fig = plt.figure()
fig.subplots_adjust(hspace=0.4, wspace=0.4)
LAP_B_ch1=np.empty((6,B_ch1.shape[0],B_ch1.shape[1]))
# A=fftpack.ifftshift(A)
for i in range(6):
    ax = fig.add_subplot(2, 3, i + 1)
    g1 = gaussian_LP(B_ch1.shape, 2 ** (i))
    B_ch1_fft = np.fft.fft2(B_ch1)
    B_ch1_fft_shift = np.fft.fftshift(B_ch1_fft)
    filtered_img1 = B_ch1_fft_shift * g1
    Lap1 = np.abs(np.fft.ifft2(np.fft.ifftshift(filtered_img1)))

    g2 = gaussian_LP(B_ch1.shape, 2 ** (i + 1))
    B_ch1_fft = np.fft.fft2(B_ch1)
    B_ch1_fft_shift = np.fft.fftshift(B_ch1_fft)
    filtered_img1 = B_ch1_fft_shift * g2
    Lap2 = np.abs(np.fft.ifft2(np.fft.ifftshift(filtered_img1)))
    sub = Lap2 - Lap1
    LAP_B_ch1[i,:,:]=sub
    # plt.imshow(sub)
    # plt.title("lap" + str(i+1))
    # plt.show()
    ax.imshow(sub)
    # plt.imshow(sub)
    plt.title("Laplacian " + str(i + 1) + " channel1")
plt.show()

fig = plt.figure()
fig.subplots_adjust(hspace=0.4, wspace=0.4)
LAP_B_ch2=np.empty((6,B_ch2.shape[0],B_ch2.shape[1]))
# A=fftpack.ifftshift(A)
for i in range(6):
    ax = fig.add_subplot(2, 3, i + 1)
    g1 = gaussian_LP(B_ch2.shape, 2 ** (i))
    B_ch2_fft = np.fft.fft2(B_ch2)
    B_ch2_fft_shift = np.fft.fftshift(B_ch2_fft)
    filtered_img1 = B_ch2_fft_shift * g1
    Lap1 = np.abs(np.fft.ifft2(np.fft.ifftshift(filtered_img1)))

    g2 = gaussian_LP(B_ch2.shape, 2 ** (i + 1))
    B_ch2_fft = np.fft.fft2(B_ch2)
    B_ch2_fft_shift = np.fft.fftshift(B_ch2_fft)
    filtered_img1 = B_ch2_fft_shift * g2
    Lap2 = np.abs(np.fft.ifft2(np.fft.ifftshift(filtered_img1)))
    sub = Lap2 - Lap1
    LAP_B_ch2[i,:,:]=sub
    # plt.imshow(sub)
    # plt.title("lap" + str(i+1))
    # plt.show()
    ax.imshow(sub)
    # plt.imshow(sub)
    plt.title("Laplacian " + str(i + 1) + " channel2")
plt.show()

fig = plt.figure()
fig.subplots_adjust(hspace=0.4, wspace=0.4)
LAP_B_ch3=np.empty((6,B_ch3.shape[0],B_ch3.shape[1]))

for i in range(6):
    ax = fig.add_subplot(2, 3, i + 1)
    g1 = gaussian_LP(B_ch3.shape, 2 ** (i))
    B_ch3_fft = np.fft.fft2(B_ch3)
    B_ch3_fft_shift = np.fft.fftshift(B_ch3_fft)
    filtered_img1 = B_ch3_fft_shift * g1
    Lap1 = np.abs(np.fft.ifft2(np.fft.ifftshift(filtered_img1)))

    g2 = gaussian_LP(B_ch3.shape, 2 ** (i + 1))
    B_ch3_fft = np.fft.fft2(B_ch3)
    B_ch3_fft_shift = np.fft.fftshift(B_ch3_fft)
    filtered_img1 = B_ch3_fft_shift * g2
    Lap2 = np.abs(np.fft.ifft2(np.fft.ifftshift(filtered_img1)))
    sub = Lap2 - Lap1
    LAP_B_ch3[i,:,:]=sub
    # plt.imshow(sub)
    # plt.title("lap" + str(i+1))
    # plt.show()
    ax.imshow(sub)
    # plt.imshow(sub)
    plt.title("Laplacian " + str(i + 1) + " channel3")
plt.show()

# ########guassian of mask

fig = plt.figure()
fig.subplots_adjust(hspace=0.4, wspace=0.4)
guassi_mask_ch1=np.empty((6,kernel_ch1.shape[0],kernel_ch1.shape[1]))
print(kernel_ch1.shape)
for i in range(6):
    ax = fig.add_subplot(2, 3, i + 1)
    g1 = gaussian_LP(kernel_ch1.shape, 2 ** (i))
    A_fft = np.fft.fft2(kernel_ch1)
    A_fft_shift = np.fft.fftshift(A_fft)
    filtered_img1 = A_fft_shift * g1
    kernel_conv = np.abs(np.fft.ifft2(np.fft.ifftshift(filtered_img1)))
    guassi_mask_ch1[i,:,:]=kernel_conv
    # plt.imshow(kernel_conv)
    # plt.title("kernel" + str(i + 1))
    # plt.show()
    ax.imshow(kernel_conv)
    # plt.imshow(sub)
    plt.title("guassian mask in level " + str(i + 1) + " channel1")
plt.show()

fig = plt.figure()
fig.subplots_adjust(hspace=0.4, wspace=0.4)
guassi_mask_ch2=np.empty((6,kernel_ch2.shape[0],kernel_ch2.shape[1]))
print(kernel_ch2.shape)
for i in range(6):
    ax = fig.add_subplot(2, 3, i + 1)
    g1 = gaussian_LP(kernel_ch2.shape, 2 ** (i))
    A_fft = np.fft.fft2(kernel_ch2)
    A_fft_shift = np.fft.fftshift(A_fft)
    filtered_img1 = A_fft_shift * g1
    kernel_conv = np.abs(np.fft.ifft2(np.fft.ifftshift(filtered_img1)))
    guassi_mask_ch2[i,:,:]=kernel_conv
    # plt.imshow(kernel_conv)
    # plt.title("kernel" + str(i + 1))
    # plt.show()
    ax.imshow(kernel_conv)
    # plt.imshow(sub)
    plt.title("guassian mask in level " + str(i + 1) + " channel2")
plt.show()

fig = plt.figure()
fig.subplots_adjust(hspace=0.4, wspace=0.4)
guassi_mask_ch3=np.empty((6,kernel_ch3.shape[0],kernel_ch3.shape[1]))
print(kernel_ch3.shape)
for i in range(6):
    ax = fig.add_subplot(2, 3, i + 1)
    g1 = gaussian_LP(kernel_ch3.shape, 2 ** (i))
    A_fft = np.fft.fft2(kernel_ch3)
    A_fft_shift = np.fft.fftshift(A_fft)
    filtered_img1 = A_fft_shift * g1
    kernel_conv = np.abs(np.fft.ifft2(np.fft.ifftshift(filtered_img1)))
    guassi_mask_ch3[i,:,:]=kernel_conv
    # plt.imshow(kernel_conv)
    # plt.title("kernel" + str(i + 1))
    # plt.show()
    ax.imshow(kernel_conv)
    # plt.imshow(sub)
    plt.title("guassian mask in level " + str(i + 1) + " channel3")
plt.show()

# ######## combining
combined_image_3=np.empty((1200,960,3))
# summation_1=np.zeros((500,620))

combine_ch1 = (guassi_mask_ch1*LAP_B_ch1+(1-guassi_mask_ch1)*LAP_A_ch1).sum(axis=0)
print(combine_ch1.shape)

combine_ch1=((combine_ch1 - combine_ch1.min()) /(combine_ch1.max() - combine_ch1.min()))

combine_ch2 = (guassi_mask_ch2*LAP_B_ch2+(1-guassi_mask_ch2)*LAP_A_ch2).sum(axis=0)
print(combine_ch2.shape)

combine_ch2=((combine_ch2 - combine_ch2.min()) /(combine_ch2.max() - combine_ch2.min()))

combine_ch3 = (guassi_mask_ch3*LAP_B_ch3+(1-guassi_mask_ch3)*LAP_A_ch3).sum(axis=0)
print(combine_ch3.shape)

combine_ch3=((combine_ch3 - combine_ch3.min()) /(combine_ch3.max() - combine_ch3.min()))


# result=np.sum(combined_image_3,axis=0)
plt.imshow(combine_ch1,cmap='gray')
plt.show()

plt.imshow(combine_ch2,cmap='gray')
plt.show()

plt.imshow(combine_ch3,cmap='gray')
plt.show()
# #
combined_image_3[:,:,0]=combine_ch1
combined_image_3[:,:,1]=combine_ch2
combined_image_3[:,:,2]=combine_ch3

# combined_image_3 = skimage.img_as_float(combined_image_3)
print("c",combined_image_3)
print(np.min(combined_image_3))

norm_image = cv.normalize(combined_image_3, None, alpha = 0, beta = 255, norm_type = cv.NORM_MINMAX, dtype = cv.CV_32F)

norm_image = norm_image.astype(np.uint8)
print("A",A)


plt.imshow(norm_image)
plt.show()


