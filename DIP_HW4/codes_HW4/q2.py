import cv2
import time
import matplotlib.pyplot as plt

img1 = cv2.imread('rouhani.png')
img2 = cv2.imread('raisi.png')

plt.imshow(cv2.cvtColor(img1,cv2.COLOR_BGR2RGB))
plt.show()

# img = cv2.imread('nasir_and_wives.png',0)

plt.imshow(cv2.cvtColor(img2,cv2.COLOR_BGR2RGB))
plt.show()



for i in range(1,11):
    alpha = (1/10)*i

    dst = cv2.addWeighted(img2, alpha, img1, 1 - alpha, 0)
    plt.imshow(cv2.cvtColor(dst,cv2.COLOR_BGR2RGB))
    plt.title("image "+str(i))
    plt.show()

#
# from PIL import Image
# from pylab import *
#
# im = array(Image.open('rouhani.png'))
# imshow(im)
# print('Please click 26 point')
# x = ginput(30,timeout=-1)
# print('you clicked:', x)
# # show()
#
# # f = open("x.txt", "w")
# with open('x.txt', 'w') as fp:
#     fp.write('\n'.join('%s %s' % x for x in x))
# fp.close()
# im = array(Image.open('raisi.png'))
# imshow(im)
# print('Please click 3 point')
# y = ginput(30,timeout=-1)
# print('you clicked:', y)
#
# with open('y.txt', 'w') as fp:
#     fp.write('\n'.join('%s %s' % x for x in y))
# fp.close()
# # f.write(y)
# # f.close()