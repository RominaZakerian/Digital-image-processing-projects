import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import random
import os, sys
from PIL import Image
import cv2

im = cv2.imread('wheres_wally_4.jpg')
print(im)
print(im.shape)
plt.imshow(cv2.cvtColor(im, cv2.COLOR_BGR2RGB))
plt.show()


im1=np.zeros(((im.shape[0],im.shape[1],3)))

for i in range(im.shape[0]):
    for j in range(im.shape[1]):
        # print(im[i,j])
        is_between1=im[i,j,0] in range(20,89)
        is_between2=im[i,j,1] in range(0,70)
        is_between3=im[i,j,2] in range(170,255)
        if i+5<im.shape[0] and i-5>0:
            is_between4=((im[i+4,j,0] in range(219,255)) or (im[i-4, j, 0] in range(219, 255)))
            is_between5=(im[i+4,j,1] in range(219,255) or (im[i-4, j, 1] in range(219, 255)))
            is_between6=(im[i+4,j,2] in range(219,255) or (im[i-4, j, 2] in range(219, 255)))
        else:
            is_between4=False
            is_between5=False
            is_between6=False

        if (is_between1 and is_between2 and is_between3):
            if (is_between4 and is_between5 and is_between6):
                im1[i,j]=im[i,j]
                print("heloo")
                for x in range(100):
                    for y in range(100):
                        if i+100<im.shape[0] and i-100>0 and j+100<im.shape[1] and j-100>0:
                            im1[i + x, j + y] = im[i + x, j + y]
                            im1[i - x, j - y] = im[i - x, j - y]
                            im1[i - 2*x, j - y] = im[i - 2*x, j - y]
                        else:
                            x=10
                            y=10
                            im1[i + x, j + y] = im[i + x, j + y]
                            im1[i - x, j - y] = im[i - x, j - y]

    if i==2300:
        print("wait:)))")#

        # else:
        #     im[i,j]=0

plt.imshow(cv2.cvtColor(im1.astype(np.uint8), cv2.COLOR_BGR2RGB))
plt.show()

