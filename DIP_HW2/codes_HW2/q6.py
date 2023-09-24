import numpy as np
import matplotlib.pyplot as plt
from skimage import io
import cv2
import glob
import math
from matplotlib.pyplot import cm
###### picture 1 ###########
f=glob.glob("frames/*.png")
sum_of_data=0
n=20
# img=cv2.imread("football_00001.png")
# data=np.asarray(img)
# print(data)
# plt.imshow(cv2.cvtColor(img,cv2.COLOR_BGR2RGB))
# plt.show()
frames=[]
for img in range(n):
    image = cv2.imread(f[img])
    data = np.asarray(image)
    # img_gray = io.imread(f[img], as_gray=True)
    # img_gray = cv2.cvtColor(data, cv2.COLOR_BGR2GRAY)
    # print(img_gray)
    # print(img_gray.shape)
    # plt.imshow(img_gray,cmap=cm.gray)
    # # print(data)
    # # print(data.shape)
    # plt.title("original image")
    # plt.show()
    # plt.figure()
    # sum_of_data=sum_of_data+image

    frames.append(data)

# print(sum_of_data)
# average=(sum_of_data/n)#.astype(dtype=np.uint8)
mean_frame = np.mean(([frame for frame in frames]), axis=0)
plt.imshow(mean_frame.astype(np.uint8))
plt.title("avereged_image")
plt.show()
# cv2.imshow("average_image",average)
# cv2.waitKey(0)
####### sunbtracting ###########

img=cv2.imread("football_test.png")
data_test=np.asarray(img)
print(data_test)
plt.imshow(cv2.cvtColor(img,cv2.COLOR_BGR2RGB))
plt.title("test_image")
plt.show()
img_gray_test = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
plt.imshow(img_gray_test,cmap='gray')
plt.title("test_image")
plt.figure()
# plt.show()
# data_test=np.uint8(img_gray_test)
# average=np.uint8(average)
# data_test=np.int32(data_test)
# median_frame=np.int32(median_frame)
sub=abs(img - mean_frame)

plt.imshow(sub.astype("uint8"))

plt.title("subtracted_image")
# plt.show()
plt.figure()
print("sub",sub)
print("sub shape",sub.shape)
# for i in range(sub.shape[0]):
#     for j in range(sub.shape[1]):
#         if ((sub[i,j]).all()>50):
#             sub[i,j]=[255,255,255]
#         else:
#             sub[i,j]=[0,0,0]
ret, thresh = cv2.threshold(sub, 0, 200, cv2.THRESH_TOZERO)
plt.imshow(thresh.astype("uint8"))

plt.title("subtracted_image")
plt.show()
# sub_gray = cv2.cvtColor(sub, cv2.COLOR_BGR2GRAY)
# sub_gray = io.imread(sub.astype('uint8'), as_gray=True)
# sub=cv2.imread(sub)
# sub_gray = cv2.cvtColor(sub, cv2.COLOR_BGR2GRAY)
# print(sub_gray)


############### median ################
# n=20
# frames=[]
# for img in range(n):
#     image = cv2.imread(f[img])
#     data = np.asarray(image)
#     plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
#     # print(data)
#     # print(data.shape)
#     plt.title("original image")
#     # plt.show()
#     # plt.figure()
#     # median=np.median(data)
#     # print(median)
#     frames.append(data)
#
# median_frame=np.median(frames,axis=0).astype(dtype=np.uint8)
# cv2.imshow('median_frame', median_frame)
# cv2.waitKey(0)

# ####### sunbtracting ###########

# img=cv2.imread("football_test.png")
# data_test=np.asarray(img)
# print(data_test)
# plt.imshow(cv2.cvtColor(img,cv2.COLOR_BGR2RGB))
# plt.title("test_image")
# plt.show()
# data_test=np.int32(data_test)
# median_frame=np.int32(median_frame)
# sub=np.abs(data_test-median_frame)
# plt.imshow(sub.astype(np.uint8))
# plt.title("after subtracting")
# plt.figure()
# cv2.imshow("sub_1",sub.astype(np.uint8))
# cv2.waitKey(0)
# plt.title("subtracted_image")
plt.show()
# sub=sub.astype(dtype=np.uint8)
#
# for i in range(sub.shape[0]):
#     for j in range(sub.shape[1]):
#         print(sub[i,j])
#         if (((sub[i,j]).all())<20):
#             print("jkbjhbhn,")
#             sub[i,j]=[0,0,0]
#         else:
#             print("hhvjhc")
# cv2.imshow("sub", sub.astype(np.uint8))
# cv2.waitKey(0)
# plt.imshow(sub.astype(dtype=np.uint8))
# # # cv2.waitKey(0)
# plt.show()