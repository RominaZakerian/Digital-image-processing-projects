import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from PIL import Image
import cv2

###### picture 1 ###########
img = mpimg.imread('color_saturation_illusion.png')
data=np.asarray(img)
imgplot = plt.imshow(img)
print(data)
print(data.shape)
plt.figure()
plt.show()
a_file = open("test.txt", "w")
for row in data:
    np.savetxt(a_file, row)

a_file.close()


for i in range(600):
    for j in range(1200):
        for k in range(3):
            if data[i,j,2]==0.7725490331649780273:
                data[i, j, 0] = 0.5882353186607360840
                data[i,j,1]=0.5882353186607360840
                data[i, j, 2] = 0.7725490331649780273
            else:
                    data[i, j, k] = 0.7058823704

plt.imshow(data)
plt.show()

################## picture 2 ###############

im = cv2.imread('gradient_optical_illusion.png')
print(im)
print(im.shape)
plt.imshow(cv2.cvtColor(im, cv2.COLOR_BGR2RGB))
plt.show()
a_file = open("test1.txt", "w")
for row in im:
    np.savetxt(a_file, row)

a_file.close()
# plt.figure()
# plt.show()
im1=np.zeros((im.shape[0],im.shape[1],3))

x=im[int(im.shape[0]/2),int(im.shape[1]/2)]
# print(im[372,512])
# for i in range()
for i in range(im.shape[0]):
    for j in range(im.shape[1]):
        if (im[i,j]==x).all():
            im1[i,j]=x

        else:
            im1[i,j]=[0,0,0]

# img = Image.fromarray(converted)
# data1=np.zeros((744,1024,3))
plt.imshow((im1).astype(np.uint8))
plt.show()

############ picture 3 #####################

img = mpimg.imread('ebbinghaus_illusion.png')
data=np.asarray(img)
imgplot = plt.imshow(img)
print(data)
print(data.shape)
# plt.figure()

a_file = open("test2.txt", "w")
for row in data:
    np.savetxt(a_file, row)

a_file.close()
plt.imshow(data)
plt.figure()


# plt.show()

for i in range(400):
    for j in range(650):
        for k in range(4):
            if data[i,j,0]==0.5686274766921997070 or data[i,j,0]==0.5764706134796142578 or data[i,j,0]==0.5607843399047851562 \
                    or data[i,j,0]==0.5647059082984924316 or data[i,j,0]==1 or data[i,j,0]==0.5725490450859069824 \
                    or data[i,j,0]==0.6666666865348815918 or data[i,j,0]==0.5019608139991760254 :
                data[i, j, 0] = 0
                data[i,j,1]=0
                data[i, j, 2] = 0
                data[i, j, 3] = 0

            # else:
            #         data[i, j, k] = 0.7058823704

plt.imshow(data)
plt.show()


############ picture 4 #####################

img = mpimg.imread('checker_shadow_illusion.png')
data=np.asarray(img)
imgplot = plt.imshow(img)
print(data)
print(data.shape)
# plt.figure()

a_file = open("test3.txt", "w")
for row in data:
    np.savetxt(a_file, row)

a_file.close()
plt.imshow(data)
plt.figure()
# plt.show()

for i in range(500 , 721):

    data[i,:]=0

for j in range(600 , 948):
    data[: , j] = 0

for i in range(383):
    data[:,i] = 0

a_file = open("test3.txt", "w")
for row in data:
    np.savetxt(a_file, row)

a_file.close()
for i in range(721):
    for j in range(948):
        if data[i,j,0]!=0.4352941215038299561 :
            data[i,j,0]=0
            data[i, j, 1] = 0
            data[i, j, 2] = 0
            data[i, j, 3] = 0
#         else:
#             data[i,j,:]=0



plt.imshow(data)
plt.show()
