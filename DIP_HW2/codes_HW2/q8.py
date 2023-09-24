import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from PIL import Image
import cv2
import math
from skimage import io
from matplotlib.pyplot import cm
from skimage.metrics import structural_similarity

# ###### picture 1 ###########
img = cv2.imread('aush.png')
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
# cv2.imshow("gray_image",img_gray)
# # cv2.waitKey(0)
img_gray = io.imread('aush.png', as_gray=True)

cv2.imshow("gray",img_gray)
cv2.waitKey(0)
# print("this is img_gray",img_gray)
# plt.imshow(img_gray)
#
# plt.title("gray_image")
# plt.show()

noise = np.random.normal(0, .01, img_gray.shape)
new_img = img_gray + noise
print(new_img)
cv2.imshow("new_image",new_img)
cv2.waitKey(0)

I=new_img
P=new_img
kernel = np.ones((8,8),np.float32)/64
mean_I = cv2.filter2D(I,-1,kernel)
mean_P = cv2.filter2D(P,-1,kernel)
# plt.subplot(121),plt.imshow(img_gray),plt.title('Original')
# plt.xticks([]), plt.yticks([])
# plt.subplot(122),plt.imshow(mean_I),plt.title('mean_I')
# plt.xticks([]), plt.yticks([])
# plt.show()

# plt.imshow(mean_P)
# plt.title("mean_p")
# plt.show()

I_I=np.multiply(I,I)
I_P=np.multiply(I,P)

corr_I = cv2.filter2D(I_I,-1,kernel)
corr_P = cv2.filter2D(I_P,-1,kernel)

var_I=corr_I - (np.multiply(mean_I,mean_I))
cov_IP=corr_P - (np.multiply(mean_I,mean_P))

a=np.divide(cov_IP,(var_I+(0.1**2)))
b=mean_P - np.multiply(a,mean_I)

mean_a=cv2.filter2D(a,-1,kernel)
mean_b=cv2.filter2D(b,-1,kernel)

q=np.multiply(mean_a,I)+mean_b
# print("this is new_imge",img_gray)
# print("this is q",q)
cv2.imshow("noisy_image",new_img)
cv2.imshow("filtered image",q)
# cv2.waitKey(0)
print(cv2.PSNR(new_img,q))
(score, diff) = structural_similarity(new_img, q, full=True)
diff = (diff * 255).astype("uint8")
cv2.namedWindow("diff", cv2.WINDOW_NORMAL)
cv2.imshow("diff", diff)
print("SSIM:{}".format(score))
cv2.waitKey(0)

# ######## noise free image as guided image ########

I=img_gray
P=new_img
kernel = np.ones((8,8),np.float32)/64
mean_I = cv2.filter2D(I,-1,kernel)
mean_P = cv2.filter2D(P,-1,kernel)
# plt.subplot(121),plt.imshow(img_gray),plt.title('Original')
# plt.xticks([]), plt.yticks([])
# plt.subplot(122),plt.imshow(mean_I),plt.title('mean_I')
# plt.xticks([]), plt.yticks([])
# plt.show()

# plt.imshow(mean_P)
# plt.title("mean_p")
# plt.show()

I_I=np.multiply(I,I)
I_P=np.multiply(I,P)

corr_I = cv2.filter2D(I_I,-1,kernel)
corr_P = cv2.filter2D(I_P,-1,kernel)

var_I=corr_I - (np.multiply(mean_I,mean_I))
cov_IP=corr_P - (np.multiply(mean_I,mean_P))

a=np.divide(cov_IP,(var_I+(0.1**2)))
b=mean_P - np.multiply(a,mean_I)

mean_a=cv2.filter2D(a,-1,kernel)
mean_b=cv2.filter2D(b,-1,kernel)

q=np.multiply(mean_a,I)+mean_b
# print("this is new_imge",img_gray)
# print("this is q",q)
cv2.imshow("gray_image",img_gray)
cv2.imshow("filtered image",q)
# cv2.waitKey(0)
print(cv2.PSNR(img_gray,q))
(score, diff) = structural_similarity(img_gray, q, full=True)
diff = (diff * 255).astype("uint8")
cv2.namedWindow("diff", cv2.WINDOW_NORMAL)
cv2.imshow("diff", diff)
print("SSIM:{}".format(score))
cv2.waitKey(0)


# # ########### apply Gaussian smoothing filter to the noisy image(part d) ##########
#
blur = cv2.GaussianBlur(new_img,(3,3),0.1)
cv2.imshow("blur",blur)
cv2.imshow("new_img",new_img)
print(cv2.PSNR(new_img,blur))
(score, diff) = structural_similarity(new_img, blur, full=True)
diff = (diff * 255).astype("uint8")
cv2.namedWindow("diff", cv2.WINDOW_NORMAL)
cv2.imshow("diff", diff)
print("SSIM:{}".format(score))
cv2.waitKey(0)

##### picture 2 ###########
img = cv2.imread('date.png')
data=np.asarray(img)
imgplot = plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
print(data)
print(data.shape)
plt.title("original image")
# plt.figure()
plt.show()

img_gray = io.imread('date.png', as_gray=True)

cv2.imshow("gray",img_gray)
cv2.waitKey(0)
print("this is img_gray",img_gray)

# noise = np.random.normal(0, .01, img_gray.shape)
# new_img = img_gray + noise
# print(new_img)
# cv2.imshow("new_image",new_img)
# cv2.waitKey(0)

I=img
P=img
kernel = np.ones((8,8),np.float32)/64
mean_I = cv2.filter2D(I,-1,kernel)
mean_P = cv2.filter2D(P,-1,kernel)
# plt.subplot(121),plt.imshow(img_gray),plt.title('Original')
# plt.xticks([]), plt.yticks([])
# plt.subplot(122),plt.imshow(mean_I),plt.title('mean_I')
# plt.xticks([]), plt.yticks([])
# plt.show()

# plt.imshow(mean_P)
# plt.title("mean_p")
# plt.show()

I_I=np.multiply(I,I)
I_P=np.multiply(I,P)

corr_I = cv2.filter2D(I_I,-1,kernel)
corr_P = cv2.filter2D(I_P,-1,kernel)

var_I=corr_I - (np.multiply(mean_I,mean_I))
cov_IP=corr_P - (np.multiply(mean_I,mean_P))

a=np.divide(cov_IP,(var_I+(0.4**2)))
b=mean_P - np.multiply(a,mean_I)

mean_a=cv2.filter2D(a,-1,kernel)
mean_b=cv2.filter2D(b,-1,kernel)

q=np.multiply(mean_a,I)+mean_b
# print("this is new_imge",img_gray)
# print("this is q",q)
cv2.imshow("original_image",img)
cv2.imshow("filtered image",q)
cv2.waitKey(0)


def psnr(img1, img2):
    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        return 100
    PIXEL_MAX = 255.0
    return 20 * math.log10(PIXEL_MAX / math.sqrt(mse))


d = psnr(img, q)
print(d)
# print(cv2.PSNR(img,q))
(score, diff) = structural_similarity(img, q, full=True , multichannel=True)
diff = (diff*255 ).astype("uint8")
cv2.namedWindow("diff", cv2.WINDOW_NORMAL)
cv2.imshow("diff", diff)
print("SSIM:{}".format(score))
cv2.waitKey(0)

########### picture 3 ###########

img = cv2.imread('zoolbia_bamieh.png')
data=np.asarray(img)
imgplot = plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
print(data)
print(data.shape)
plt.title("original image")
# plt.figure()
plt.show()

img_gray = io.imread('zoolbia_bamieh.png', as_gray=True)

# cv2.imshow("gray",img_gray)
# cv2.waitKey(0)
plt.imshow(img_gray,cmap=cm.gray)
plt.show()
print("this is img_gray",img_gray)


I=img
P=img
kernel = np.ones((4,4),np.float32)/16
mean_I = cv2.filter2D(I,-1,kernel)
mean_P = cv2.filter2D(P,-1,kernel)


I_I=np.multiply(I,I)
I_P=np.multiply(I,P)

corr_I = cv2.filter2D(I_I,-1,kernel)
corr_P = cv2.filter2D(I_P,-1,kernel)

var_I=corr_I - (np.multiply(mean_I,mean_I))
cov_IP=corr_P - (np.multiply(mean_I,mean_P))

a=np.divide(cov_IP,(var_I+(0.2**2)))
b=mean_P - np.multiply(a,mean_I)

mean_a=cv2.filter2D(a,-1,kernel)
mean_b=cv2.filter2D(b,-1,kernel)

q=np.multiply(mean_a,I)+mean_b
# print("this is new_imge",img_gray)
print("this is q",q)
# cv2.imshow("original_image",img)
# cv2.imshow("filtered image",q)
# cv2.waitKey(0)

d=P-q

alpha=5
p_enhanced=q+alpha*d

plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
plt.title("original image")
plt.figure()
plt.imshow(cv2.cvtColor(q.astype('float32'), cv2.COLOR_BGR2RGB))
plt.title("filtered image")
plt.show()
plt.imshow(cv2.cvtColor(p_enhanced.astype('float32'), cv2.COLOR_BGR2RGB))
plt.title("enhanced image")
plt.show()
# cv2.imshow("original_image",img)
# cv2.imshow("filtered_image",q)
# cv2.imshow("enhanced image",p_enhanced)
# cv2.waitKey(0)




