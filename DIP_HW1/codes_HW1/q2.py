import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from mpl_toolkits.axes_grid1 import ImageGrid
import cv2
from PIL import Image


img = plt.imread('10.jpg')

plt.imshow(img)

data=np.asarray(img)

print(data)
print(data.shape)

a_file = open("test4.txt", "w")
for row in data:
    np.savetxt(a_file, row)

a_file.close()

######## part a #########
def extract_channels():

    data1 = np.empty((341, 392))


    data2 = np.empty((341, 392))
    data3 = np.empty((341, 392))
    for i in range(data1.shape[0]):
        for j in range(data1.shape[1]):
            data1[i, j] = data[i, j]


    for i in range(data2.shape[0]):
        for j in range(data2.shape[1]):
            data2[i, j] = data[i + 341, j]

    # axarr[2].imshow(data2)

    for i in range(data3.shape[0]):
        for j in range(data3.shape[1]):
            data3[i, j] = data[i + 682, j]


    return data1,data2,data3

data1,data2,data3=extract_channels()
fig = plt.figure(figsize=(4., 4.))
grid = ImageGrid(fig, 111,  # similar to subplot(111)
                 nrows_ncols=(3, 1),  # creates 2x2 grid of axes
                 axes_pad=0.3,  # pad between axes in inch.
                 )

for ax, im in zip(grid, [data1 , data2, data3]):
    # Iterating over the grid returns the Axes.
    ax.imshow(im)

grid[0].set_title("Blue", fontdict=None, loc='left', color = "k")
grid[1].set_title("Green", fontdict=None, loc='left', color = "k")
grid[2].set_title("Red", fontdict=None, loc='left', color = "k")
plt.show()

########## part b ############
def stack_channels():

    rgb = np.dstack((data3,data2,data1))
    print(rgb.shape)
    return rgb

RGB=stack_channels()
plt.imshow((RGB ).astype(np.uint8))
# plt.show()

########## part c ##############
def metric(a,b):
    a=a-a.mean(axis=0)
    b=b-b.mean(axis=0)
    return np.sum(((a/np.linalg.norm(a)) * (b/np.linalg.norm(b))))

def align_resultss(a, b, t):
    min_ncc = -1
    ivalue=np.linspace(-t,t,2*t,dtype=int)
    jvalue=np.linspace(-t,t,2*t,dtype=int)
    for i in ivalue:
        for j in jvalue:
            nccDiff = metric(a,np.roll(b,[i,j],axis=(0,1)))
            if nccDiff > min_ncc:
                min_ncc = nccDiff
                output = [i,j]

    return output
alignGtoB = align_resultss(data1,data2,30)
alignRtoB = align_resultss(data1,data3,30)
print(alignGtoB, alignRtoB)
g=np.roll(data2,alignGtoB,axis=(0,1))
r=np.roll(data3,alignRtoB,axis=(0,1))
coloured_origin = (np.dstack((r,g,data1))).astype(np.uint8)
coloured_origin=coloured_origin[int(coloured_origin.shape[0]*0.05):int(coloured_origin.shape[0]-coloured_origin.shape[0]*0.05),int(coloured_origin.shape[1]*0.05):int(coloured_origin.shape[1]-coloured_origin.shape[1]*0.05)]
coloured_origin = Image.fromarray(coloured_origin)

plt.figure()
plt.imshow(coloured_origin)
plt.show()

############ part d ############

######## channel 1 ##########
scale_percent = 50

#calculate the 50 percent of original dimensions
width = int(data1.shape[1] * scale_percent / 100)
height = int(data1.shape[0] * scale_percent / 100)

# dsize
dsize = (width, height)

# resize image
output1 = cv2.resize(data1, dsize)

plt.imshow(output1)
# plt.show()
plt.figure()
plt.imshow(data1)
plt.show()

######## channel 2 ########
scale_percent = 50

#calculate the 50 percent of original dimensions
width = int(data2.shape[1] * scale_percent / 100)
height = int(data2.shape[0] * scale_percent / 100)

# dsize
dsize = (width, height)

# resize image
output2 = cv2.resize(data2, dsize)

plt.imshow(output2)
# plt.show()
plt.figure()
plt.imshow(data2)
plt.show()

######### channel 3 #######
scale_percent = 50

#calculate the 50 percent of original dimensions
width = int(data3.shape[1] * scale_percent / 100)
height = int(data3.shape[0] * scale_percent / 100)

# dsize
dsize = (width, height)

# resize image
output3 = cv2.resize(data3, dsize)

plt.imshow(output3)
# plt.show()
plt.figure()
plt.imshow(data3)
plt.show()

alignGtoB = align_resultss(output1,output2,15)
alignRtoB = align_resultss(output1,output3,15)
print(alignGtoB, alignRtoB)
print(alignGtoB[0])
print(alignGtoB[1])

g=np.roll(output2,alignGtoB,axis=(0,1))
r=np.roll(output3,alignRtoB,axis=(0,1))
coloured = (np.dstack((r,g,output1))).astype(np.uint8)
coloured=coloured[int(coloured.shape[0]*0.05):int(coloured.shape[0]-coloured.shape[0]*0.05),int(coloured.shape[1]*0.05):int(coloured.shape[1]-coloured.shape[1]*0.05)]
coloured = Image.fromarray(coloured)

# plt.figure()
plt.imshow(coloured)
plt.show()

def align_results1(a, b, t,x,y):
    min_ncc = -1
    ivalue=np.linspace(-t,t,2*t,dtype=int)
    jvalue=np.linspace(-t,t,2*t,dtype=int)
    for i in ivalue:
        for j in jvalue:
            nccDiff = metric(a,np.roll(b,[i+x,j+y],axis=(0,1)))
            if nccDiff > min_ncc:
                min_ncc = nccDiff
                output = [i+x,j+y]

    return output

alignGtoB1 = align_results1(data1,data2,15,alignGtoB[0],alignGtoB[1])
alignRtoB1 = align_results1(data1,data3,15,alignRtoB[0],alignRtoB[1])
print(alignGtoB, alignRtoB)
# print(alignGtoB[0])
# print(alignGtoB[1])

g=np.roll(data2,alignGtoB1,axis=(0,1))
r=np.roll(data3,alignRtoB1,axis=(0,1))
coloured1 = (np.dstack((r,g,data1))).astype(np.uint8)
coloured1=coloured1[int(coloured1.shape[0]*0.05):int(coloured1.shape[0]-coloured1.shape[0]*0.05),int(coloured1.shape[1]*0.05):int(coloured1.shape[1]-coloured1.shape[1]*0.05)]
coloured1 = Image.fromarray(coloured1)

# plt.figure()
plt.imshow(coloured_origin)
# plt.show()
plt.figure()
plt.imshow(coloured1)
plt.show()
