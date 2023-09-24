import numpy as np
import matplotlib.pyplot as plt
from skimage import io
import cv2
import math
from matplotlib.pyplot import cm
import scipy
from scipy import signal

###### picture 1 ###########
img = cv2.imread('pavement_1.png')
data=np.asarray(img)
imgplot = plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
print(data)
print(data.shape)
plt.title("original image")
plt.figure()
# plt.show()

img1 = cv2.imread('pavement_2.png')
data1=np.asarray(img1)
plt.imshow(cv2.cvtColor(img1, cv2.COLOR_BGR2RGB))
print(data1)
print(data1.shape)
plt.title("original image_2")
# plt.figure()
plt.show()

# img_gray = io.imread('highway_1.png', as_gray=True)
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
print(img_gray)
print(img_gray.shape)
plt.imshow(img_gray,cmap=cm.gray)
plt.title("original image _1 gray_scale")
plt.figure()

# plt.show()

img_gray_2 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
print(img_gray_2)
print(img_gray_2.shape)
plt.imshow(img_gray_2,cmap=cm.gray)
plt.title("original image _2 gray_scale")
plt.show()
array=np.empty((img_gray_2.shape[0],img_gray_2.shape[1],4))
for k in range(8):
    # k=0
    result=(np.floor(img_gray/(2**k)))%2
    print(result)

    plt.imshow(result,cmap=cm.gray)
    plt.title(" %ith bit_plane"% (k+1))
    # plt.show()
    plt.figure()

    result2=(np.floor(img_gray_2/(2**k)))%2
    print(result2)
    plt.imshow(result2,cmap=cm.gray)
    plt.title("%ith bit_plane for seconed image" % (k + 1))
    plt.figure()
    # plt.show()

    XOR=np.logical_xor(result,result2)
    plt.imshow(XOR,cmap=cm.gray)
    plt.title("XOR for  %i th bit_plane"% (k+1))
    plt.show()
    if k>=4:
        array[:,:,k-4]=XOR
print(array)
print(array.shape)
final_result=0
for k in range(4):
    temp=(2**k)*array[:,:,k]
    final_result=final_result+temp
plt.imshow(final_result,cmap='gray')
plt.title("final_image")
plt.show()

result1 = signal.medfilt2d(final_result,[3,3])

plt.imshow(result1,cmap='gray')
plt.title("final_image after enhancing")
plt.show()







