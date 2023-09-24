import csv
import cv2
import os, sys
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import random
from subprocess import Popen, PIPE
from PIL import Image
from scipy.spatial import Delaunay
import argparse
from subprocess import Popen, PIPE

# plt.imshow('rouhani.png')
# lines = []

rouhani_points=np.empty((30,2),dtype=np.float64)

with open('x.txt', newline='\n') as f:
    reader = csv.reader(f, delimiter=' ')
    # print(len(reader))
    j=-1
    for i in reader:
        # print(i)
        j+=1
        A = [float(x) for x in i]
        # print(A)
        # for x in A:
        rouhani_points[j][0] = A[0]
        rouhani_points[j][1] = A[1]

print("rouhani_points:\n",rouhani_points)

raisi_points=np.empty((30,2),dtype=np.float64)

with open('y.txt', newline='\n') as f:
    reader = csv.reader(f, delimiter=' ')
    # print(len(reader))
    j=-1
    for i in reader:
        # print(i)
        j+=1
        A = [float(x) for x in i]
        # print(A)
        # for x in A:
        raisi_points[j][0] = A[0]
        raisi_points[j][1] = A[1]

print("raisi_points:\n",raisi_points)
        # print(len(i))

# with open('x.txt') as f:
#     contents = f.readlines()
#
# print(contents)
image1 = plt.imread('rouhani.png')
print(image1.shape)
if len(image1.shape) > 2 and image1.shape[2] == 4:
    #convert the image from RGBA2RGB
    image1 = cv2.cvtColor(image1, cv2.COLOR_BGRA2BGR)
plt.imshow(image1)
# plt.figure()
# plt.show()
for i in range(30):

    plt.plot(rouhani_points[i][0], rouhani_points[i][1], 'ro')
#
plt.show()

image2 = plt.imread('raisi.png')
print(image2.shape)

if len(image2.shape) > 2 and image2.shape[2] == 4:
    #convert the image from RGBA2RGB
    image2 = cv2.cvtColor(image2, cv2.COLOR_BGRA2BGR)

plt.imshow(image2)
# plt.figure()
for i in range(30):

    plt.plot(raisi_points[i][0], raisi_points[i][1], 'ro')
#
plt.show()


def applyAffineTransform(src, srcTri, dstTri, size):
    # Given a pair of triangles, find the affine transform.
    warpMat = cv2.getAffineTransform(np.float32(srcTri), np.float32(dstTri))

    # Apply the Affine Transform just found to the src image
    dst = cv2.warpAffine(src, warpMat, (size[0], size[1]), None, flags=cv2.INTER_LINEAR,
                         borderMode=cv2.BORDER_REFLECT_101)

    return dst


tri_rouhani = Delaunay(rouhani_points)

image = plt.imread('rouhani.png')
plt.imshow(image)
plt.triplot(rouhani_points[:,0], rouhani_points[:,1], tri_rouhani.simplices)
plt.plot(rouhani_points[:,0], rouhani_points[:,1], 'o')
plt.show()


tri_raisi = Delaunay(raisi_points)

image = plt.imread('raisi.png')
plt.imshow(image)
plt.triplot(raisi_points[:,0], raisi_points[:,1], tri_rouhani.simplices)
plt.plot(raisi_points[:,0], raisi_points[:,1], 'o')
plt.show()

totalImages=int(0.5*60)

theResult=np.zeros(image1.shape, dtype=image1.dtype)
# p = Popen(['ffmpeg', '-y', '-f', 'image2pipe', '-r', str(60),'-s',str(885)+'x'+str(709), '-i', '-', '-c:v', 'libx264', '-crf', '25','-vf','scale=trunc(iw/2)*2:trunc(ih/2)*2','-pix_fmt','yuv420p', theResult], stdin=PIPE)
for j in range(0,10):
    morphed_points=np.empty((30,2),dtype=np.float64)

    alpha = j / (11 - 1)

    for i in range(30):
        morphed_points[i][0]=(1-alpha)*rouhani_points[i][0]+alpha*raisi_points[i][0]
        morphed_points[i][1] = (1 - alpha) * rouhani_points[i][1] + alpha * raisi_points[i][1]


    tri_morphed=Delaunay(morphed_points)

    # plt.triplot(morphed_points[:,0], morphed_points[:,1], tri_rouhani.simplices)
    # plt.plot(morphed_points[:,0], morphed_points[:,1], 'o')
    # plt.show()

    print("indices of triangle\n",tri_rouhani.simplices)

    print((tri_rouhani.simplices).shape)



    imgMorph = np.zeros(image1.shape, dtype=image1.dtype)
    for i in range(54):
        indice1=tri_rouhani.simplices[i][0]
        indice2=tri_rouhani.simplices[i][1]
        indice3=tri_rouhani.simplices[i][2]
        input_tri = np.empty((3, 2), dtype=np.float32)
        input_tri[0][:]=rouhani_points[indice1][:]
        input_tri[1][:]=rouhani_points[indice2][:]
        input_tri[2][:]=rouhani_points[indice3][:]

        print("iput_tri\n",input_tri)

        output_tri=np.empty((3,2),dtype=np.float32)

        output_tri[0][:]=raisi_points[indice1][:]
        output_tri[1][:]=raisi_points[indice2][:]
        output_tri[2][:]=raisi_points[indice3][:]

        print("output_tri\n",output_tri)

        morphed_tri = np.empty((3, 2), dtype=np.float32)

        morphed_tri[0][:] = morphed_points[indice1][:]
        morphed_tri[1][:] = morphed_points[indice2][:]
        morphed_tri[2][:] = morphed_points[indice3][:]

        print("output_tri\n", output_tri)


        r1 = cv2.boundingRect(np.float32([input_tri]))
        r2 = cv2.boundingRect(np.float32([output_tri]))
        r = cv2.boundingRect(np.float32([morphed_tri]))


        # Offset points by left top corner of the respective rectangles
        t1Rect = []
        t2Rect = []
        tRect = []


        for i in range(0, 3):
            tRect.append(((morphed_tri[i][0] - r[0]),(morphed_tri[i][1] - r[1])))
            t1Rect.append(((input_tri[i][0] - r1[0]),(input_tri[i][1] - r1[1])))
            t2Rect.append(((output_tri[i][0] - r2[0]),(output_tri[i][1] - r2[1])))


        # Get mask by filling triangle
        mask = np.zeros((r[3], r[2], 3), dtype = np.float32)
        cv2.fillConvexPoly(mask, np.int32(tRect), (1.0, 1.0, 1.0), 16, 0)

        # Apply warpImage to small rectangular patches
        img1Rect = image1[r1[1]:r1[1] + r1[3], r1[0]:r1[0] + r1[2]]
        img2Rect = image2[r2[1]:r2[1] + r2[3], r2[0]:r2[0] + r2[2]]

        size = (r[2], r[3])
        warpImage1 = applyAffineTransform(img1Rect, t1Rect, tRect, size)
        warpImage2 = applyAffineTransform(img2Rect, t2Rect, tRect, size)

        # Alpha blend rectangular patches
        imgRect = (1.0 - alpha) * warpImage1 + alpha * warpImage2

        # Copy triangular region of the rectangular patch to the output image
        imgMorph[r[1]:r[1]+r[3], r[0]:r[0]+r[2]] = imgMorph[r[1]:r[1]+r[3], r[0]:r[0]+r[2]] * ( 1 - mask ) + imgRect * mask

        # plt.imshow(imgMorph)
        # plt.show()
    # temp_res=cv2.cvtColor(np.uint8(imgMorph),cv2.COLOR_BGR2RGB)
    # res=Image.fromarray(temp_res)
    plt.imshow(imgMorph)
    plt.title(str(j+1)+"th image")
    plt.show()
    # cv2.imshow("morphed",cv2.cvtColor(imgMorph,cv2.COLOR_BGR2RGB))
    # cv2.waitKey(0)


# size = image.shape
# rect = (0, 0, size[1], size[0])
#
# subdiv  = cv2.Subdiv2D(rect)
#
# def rect_contains(rect, point) :
#     if point[0] < rect[0] :
#         return False
#     elif point[1] < rect[1] :
#         return False
#     elif point[0] > rect[2] :
#         return False
#     elif point[1] > rect[3] :
#         return False
#     return True
#
# # Draw a point
# def draw_point(img, p, color ) :
#     cv2.circle( img, p, 2, color, cv2.FILLED, cv2.LINE_AA, 0 )
#
# # Draw delaunay triangles
#
# def draw_delaunay(img, subdiv, delaunay_color ) :
#
#     triangleList = subdiv.getTriangleList()
#     size = img.shape
#     r = (0, 0, size[1], size[0])
#     print("\n""triangle list",triangleList,"\n")
#
#     for t in triangleList :
#
#         pt1 = (t[0], t[1])
#         pt2 = (t[2], t[3])
#         pt3 = (t[4], t[5])
#         pts=[]
#         pts.append(pt1)
#         pts.append(pt2)
#         pts.append(pt3)
#         print(pts)
#         # with open('points_rouhani.txt', 'w') as fp:
#             # fp.write(''.join(format( x for x in pts)))
#         # fp.close()
#
#         if rect_contains(r, pt1) and rect_contains(r, pt2) and rect_contains(r, pt3) :
#
#             cv2.line(img, pt1, pt2, delaunay_color, 1, cv2.LINE_AA, 0)
#             cv2.line(img, pt2, pt3, delaunay_color, 1, cv2.LINE_AA, 0)
#             cv2.line(img, pt3, pt1, delaunay_color, 1, cv2.LINE_AA, 0)
#
# # Draw voronoi diagram
# def draw_voronoi(img, subdiv) :
#
#     ( facets, centers) = subdiv.getVoronoiFacetList([])
#
#     for i in range(0,len(facets)) :
#         ifacet_arr = []
#         for f in facets[i] :
#             ifacet_arr.append(f)
#
#         ifacet = np.array(ifacet_arr, np.int)
#         color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
#
#         cv2.fillConvexPoly(img, ifacet, color, cv2.LINE_AA, 0)
#         ifacets = np.array([ifacet])
#         cv2.polylines(img, ifacets, True, (0, 0, 0), 1, cv2.LINE_AA, 0)
#         cv2.circle(img, (centers[i][0], centers[i][1]), 3, (0, 0, 0), cv2.FILLED, cv2.LINE_AA, 0)
#
# if __name__ == '__main__':
#
#     # Define window names
#     win_delaunay = "Delaunay Triangulation"
#     win_voronoi = "Voronoi Diagram"
#
#     # Turn on animation while drawing triangles
#     animate = True
#
#     # Define colors for drawing.
#     delaunay_color = (255,255,255)
#     points_color = (0, 0, 255)
#
#     # Read in the image.
#     img = cv2.imread("rouhani.png")
#
#     # Keep a copy around
#     img_orig = img.copy()
#
#     # Rectangle to be used with Subdiv2D
#     size = img.shape
#     rect = (0, 0, size[1], size[0])
#
#     # Create an instance of Subdiv2D
#     subdiv = cv2.Subdiv2D(rect)
#
#     # Create an array of points.
#     points = []
#
#     # Read in the points from a text file
#     with open("x.txt") as file :
#         for line in file :
#             x, y = line.split()
#             points.append((int(float(x)), int(float(y))))
#
#     # Insert points into subdiv
#     for p in points :
#         subdiv.insert(p)
#
#         # Show animation
#         if animate :
#             img_copy = img_orig.copy()
#             # Draw delaunay triangles
#             draw_delaunay( img_copy, subdiv, (255, 255, 255) )
#             cv2.imshow(win_delaunay, img_copy)
#             cv2.waitKey(100)
#
#     # Draw delaunay triangles
#     draw_delaunay( img, subdiv, (255, 255, 255) )
#
#     # Draw points
#     for p in points :
#         draw_point(img, p, (0,0,255))
#
#     # Allocate space for Voronoi Diagram
#     # img_voronoi = np.zeros(img.shape, dtype = img.dtype)
#     #
#     # # Draw Voronoi diagram
#     # draw_voronoi(img_voronoi,subdiv)
#     #
#     # # Show results
#     cv2.imshow(win_delaunay,img)
#     # cv2.imshow(win_voronoi,img_voronoi)
#     cv2.waitKey(0)
#
#     ########### raisi image ###########
#     print("\n""raisi_image""\n")
#     win_delaunay = "Delaunay Triangulation"
#     win_voronoi = "Voronoi Diagram"
#
#     # Turn on animation while drawing triangles
#     animate = True
#
#     # Define colors for drawing.
#     delaunay_color = (255, 255, 255)
#     points_color = (0, 0, 255)
#
#     # Read in the image.
#     img = cv2.imread("raisi.png")
#
#     # Keep a copy around
#     img_orig = img.copy()
#
#     # Rectangle to be used with Subdiv2D
#     size = img.shape
#     rect = (0, 0, size[1], size[0])
#
#     # Create an instance of Subdiv2D
#     subdiv = cv2.Subdiv2D(rect)
#
#     # Create an array of points.
#     points = []
#
#     # Read in the points from a text file
#     with open("y.txt") as file:
#         for line in file:
#             x, y = line.split()
#             points.append((int(float(x)), int(float(y))))
#
#     # Insert points into subdiv
#     for p in points:
#         subdiv.insert(p)
#
#         # Show animation
#         if animate:
#             img_copy = img_orig.copy()
#             # Draw delaunay triangles
#             draw_delaunay(img_copy, subdiv, (255, 255, 255))
#             cv2.imshow(win_delaunay, img_copy)
#             cv2.waitKey(100)
#
#     # Draw delaunay triangles
#     draw_delaunay(img, subdiv, (255, 255, 255))
#
#     # Draw points
#     for p in points:
#         draw_point(img, p, (0, 0, 255))
#
#     # Allocate space for Voronoi Diagram
#     # img_voronoi = np.zeros(img.shape, dtype = img.dtype)
#     #
#     # # Draw Voronoi diagram
#     # draw_voronoi(img_voronoi,subdiv)
#     #
#     # # Show results
#     cv2.imshow(win_delaunay, img)
#     # cv2.imshow(win_voronoi,img_voronoi)
#     cv2.waitKey(0)
#

