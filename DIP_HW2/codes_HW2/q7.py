import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage import io
from scipy import ndimage
from matplotlib import cm
from scipy.ndimage.filters import convolve

img = cv2.imread('donald_plays_golf_1.png')
data=np.asarray(img)
imgplot = plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
print(data)
print(data.shape)
plt.title("original image")
# plt.figure()
plt.show()


img_gray = io.imread('donald_plays_golf_1.png', as_gray=True)

plt.imshow(img_gray,cmap='gray')
plt.show()
# cv2.imshow("gray",img_gray)
# cv2.waitKey(0)
print("this is img_gray",img_gray)
print(img_gray.shape)

kx = np.array([[1,0,-1],[2,0,-2],[1,0,-1]])
# Define kernel for y differences
ky = np.array([[1,2,1] ,[0,0,0], [-1,-2,-1]])
# Perform x convolution
x=ndimage.convolve(img_gray,kx)
# Perform y convolution
y=ndimage.convolve(img_gray,ky)
sobel=np.hypot(x,y)
print("this is gradient image",sobel)
print("sobel_shape",sobel.shape)
plt.imshow(sobel,cmap=cm.gray)
plt.title("gradient image")
plt.show()


def calc_energy(img):
    filter_du = np.array([
        [1.0, 2.0, 1.0],
        [0.0, 0.0, 0.0],
        [-1.0, -2.0, -1.0],
    ])
    # This converts it from a 2D filter to a 3D filter, replicating the same
    # filter for each channel: R, G, B
    filter_du = np.stack([filter_du] * 3, axis=2)

    filter_dv = np.array([
        [1.0, 0.0, -1.0],
        [2.0, 0.0, -2.0],
        [1.0, 0.0, -1.0],
    ])
    # This converts it from a 2D filter to a 3D filter, replicating the same
    # filter for each channel: R, G, B
    filter_dv = np.stack([filter_dv] * 3, axis=2)

    img = img.astype('float32')
    convolved = np.absolute(convolve(img, filter_du)) + np.absolute(convolve(img, filter_dv))

    # We sum the energies in the red, green, and blue channels
    energy_map = convolved.sum(axis=2)

    return energy_map

energy=calc_energy(img)
print("this is energy",energy)
print(energy.shape)
# energy=energy.reshape((energy.shape[0],energy.shape[1],3))
plt.imshow(energy,cmap=cm.gray)
plt.title("this is gradient_image")
plt.show()


def minimum_seam(img):
    r, c, _ = img.shape
    energy_map = calc_energy(img)
    # energy=np.zeros((r,c,3))

    M = energy_map.copy()
    backtrack = np.zeros_like(M, dtype=np.int)
    # print(M)

    for i in range(1, r):
        for j in range(0, c):
            # Handle the left edge of the image, to ensure we don't index -1
            if j == 0:
                idx = np.argmin(M[i - 1, j:j + 2])
                backtrack[i, j] = idx + j
                min_energy = M[i - 1, idx + j]
            else:
                idx = np.argmin(M[i - 1, j - 1:j + 2])
                backtrack[i, j] = idx + j - 1
                min_energy = M[i - 1, idx + j - 1]

            M[i, j] += min_energy



    return M, backtrack

# energy = np.zeros((img.shape[0], img.shape[1], 3))
def carve_column(img):
    r, c, _ = img.shape

    M, backtrack = minimum_seam(img)

    # Create a (r, c) matrix filled with the value True
    # We'll be removing all pixels from the image which
    # have False later
    mask = np.ones((r, c), dtype=np.bool)

    # Find the position of the smallest element in the
    # last row of M
    j = np.argmin(M[-1])

    for i in reversed(range(r)):
        # Mark the pixels for deletion
        mask[i, j] = False
        data[i,j]=[0,0,255]
        j = backtrack[i, j]

    # Since the image has 3 channels, we convert our
    # mask to 3D
    mask = np.stack([mask] * 3, axis=2)

    # Delete all the pixels marked False in the mask,
    # and resize it to the new image dimensions
    img = img[mask].reshape((r, c - 1, 3))

    return img


def crop_c(img, scale_c):
    r, c, _ = img.shape
    new_c = int(scale_c * c)

    for i in range(new_c): # use range if you don't want to use tqdm
        img = carve_column(img)
        print("hello")

    plt.imshow(cv2.cvtColor(data, cv2.COLOR_BGR2RGB))
    plt.show()

    return img

new_image=crop_c(img,0.25)
plt.imshow(cv2.cvtColor(new_image, cv2.COLOR_BGR2RGB))
plt.show()
print(img.shape)
print(new_image.shape)