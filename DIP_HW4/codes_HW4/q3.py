import numpy as np
import cv2
import matplotlib.pyplot as plt
#
# I1 = cv2.imread('aut_1.png')
# G1 = cv2.cvtColor(I1,cv2.COLOR_BGR2GRAY)
#
# I2 = cv2.imread('aut_2.png')
# G2 = cv2.cvtColor(I2,cv2.COLOR_BGR2GRAY)
#
# sift = cv2.xfeatures2d.SIFT_create() # opencv 3
# # use "sift = cv2.SIFT()" if the above fails
#
# # detect keypoints and compute their disriptor vectors
# keypoints1, desc1 = sift.detectAndCompute(G1, None) # opencv 3
# keypoints2, desc2 = sift.detectAndCompute(G2, None) # opencv 3
#
# print("No. of keypoints1 =", len(keypoints1))
# print("No. of keypoints2 =", len(keypoints2))
#
# print("Descriptors1.shape =", desc1.shape)
# print("Descriptors2.shape =", desc2.shape)
#
# # stop here!!
# # exit() # comment this line out to move on!
#
# # brute-force matching
# bf = cv2.BFMatcher()
#
# # for each descriptor in desc1 find its
# # two nearest neighbours in desc2
# matches = bf.knnMatch(desc1,desc2, k=2)
#
# good_matches = []
# alpha = 0.75
# for m1,m2 in matches:
#     # m1 is the best match
#     # m2 is the second best match
#     if m1.distance < alpha *m2.distance:
#         good_matches.append(m1)
#
# # apply RANSAC
# points1 = [keypoints1[m.queryIdx].pt for m in good_matches]
# points1 = np.array(points1,dtype=np.uint8)
#
# points2 = [keypoints2[m.trainIdx].pt for m in good_matches]
# points2 = np.array(points2,dtype=np.uint8)
# H, mask = cv2.findHomography(points1, points2, cv2.RANSAC,5.0) # 5 pixels margin
# mask = mask.ravel().tolist()
# print(mask)
#
# good_matches = [m for m,msk in zip(good_matches,mask) if msk == 1]
#
# I = cv2.drawMatches(I1,keypoints1,I2,keypoints2, good_matches, None)
#
# plt.imshow(I)
# # cv2.waitKey(0)
# plt.show()
# print("hello")
# width = I1.shape[1] + I2.shape[1]
# height = I1.shape[0] + I2.shape[0]
#
# result = cv2.warpPerspective(I1.astype(np.uint8), H, (width, height))
# # result[0:queryImg.shape[0], 0:queryImg.shape[1]] = queryImg
#
# # plt.figure(figsize=(20,10))
# plt.imshow(result)
#
# plt.axis('off')
# plt.show()
#
# # plt.savefig("aut_12.png",result)
# # cv2.imwrite("aut_12.png",result)

#################
img1 = cv2.imread("aut_1.png")
img2 = cv2.imread("aut_2.png")

img1_gray = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
img2_gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

plt.imshow(img1_gray)
plt.imshow(img2_gray)

orb = cv2.xfeatures2d.SIFT_create(nfeatures=2000)

# Find the key points and descriptors with ORB
keypoints1, descriptors1 = orb.detectAndCompute(img1, None)
keypoints2, descriptors2 = orb.detectAndCompute(img2, None)

plt.imshow(cv2.drawKeypoints(img1, keypoints1, None, (255, 0, 255)))
plt.imshow(cv2.drawKeypoints(img2, keypoints2, None, (255, 0, 255)))
# bf = cv2.BFMatcher_create(cv2.NORM_HAMMING)
bf = cv2.BFMatcher()
# Find matching points
matches = bf.knnMatch(descriptors1, descriptors2,k=2)


def draw_matches(img1, keypoints1, img2, keypoints2, matches):
    r, c = img1.shape[:2]
    r1, c1 = img2.shape[:2]

    # Create a blank image with the size of the first image + second image
    output_img = np.zeros((max([r, r1]), c + c1, 3), dtype='uint8')
    output_img[:r, :c, :] = np.dstack([img1, img1, img1])
    output_img[:r1, c:c + c1, :] = np.dstack([img2, img2, img2])

    # Go over all of the matching points and extract them
    for match in matches:
        img1_idx = match.queryIdx
        img2_idx = match.trainIdx
        (x1, y1) = keypoints1[img1_idx].pt
        (x2, y2) = keypoints2[img2_idx].pt

        # Draw circles on the keypoints
        cv2.circle(output_img, (int(x1), int(y1)), 4, (0, 255, 255), 1)
        cv2.circle(output_img, (int(x2) + c, int(y2)), 4, (0, 255, 255), 1)

        # Connect the same keypoints
        cv2.line(output_img, (int(x1), int(y1)), (int(x2) + c, int(y2)), (0, 255, 255), 1)

    return output_img

all_matches = []
for m, n in matches:
  all_matches.append(m)

img3 = draw_matches(img1_gray, keypoints1, img2_gray, keypoints2, all_matches[:30])
plt.imshow(img3)

good = []
for m, n in matches:
    if m.distance < 0.6 * n.distance:
        good.append(m)

plt.imshow(cv2.drawKeypoints(img1, [keypoints1[m.queryIdx] for m in good], None, (255, 0, 255)))
plt.imshow(cv2.drawKeypoints(img2, [keypoints2[m.trainIdx] for m in good], None, (255, 0, 255)))
def warpImages(img1, img2, H):
    rows1, cols1 = img1.shape[:2]
    rows2, cols2 = img2.shape[:2]

    list_of_points_1 = np.float32([[0, 0], [0, rows1], [cols1, rows1], [cols1, 0]]).reshape(-1, 1, 2)
    temp_points = np.float32([[0, 0], [0, rows2], [cols2, rows2], [cols2, 0]]).reshape(-1, 1, 2)

    # When we have established a homography we need to warp perspective
    # Change field of view
    list_of_points_2 = cv2.perspectiveTransform(temp_points, H)

    list_of_points = np.concatenate((list_of_points_1, list_of_points_2), axis=0)

    [x_min, y_min] = np.int32(list_of_points.min(axis=0).ravel() - 0.5)
    [x_max, y_max] = np.int32(list_of_points.max(axis=0).ravel() + 0.5)

    translation_dist = [-x_min, -y_min]

    H_translation = np.array([[1, 0, translation_dist[0]], [0, 1, translation_dist[1]], [0, 0, 1]])

    output_img = cv2.warpPerspective(img2, H_translation.dot(H), (x_max - x_min, y_max - y_min))
    output_img[translation_dist[1]:rows1 + translation_dist[1], translation_dist[0]:cols1 + translation_dist[0]] = img1

    return output_img


MIN_MATCH_COUNT = 10

if len(good) > MIN_MATCH_COUNT:
    # Convert keypoints to an argument for findHomography
    src_pts = np.float32([keypoints1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    dst_pts = np.float32([keypoints2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

    # Establish a homography
    M, _ = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

    result = warpImages(img2, img1, M)

    # cv2_imshow(result)

    plt.imshow(result)
    plt.show()
    cv2.imwrite("aut_12.png",result)

#####################################
img1 = cv2.imread("aut_12.png")
img2 = cv2.imread("aut_3.png")

img1_gray = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
img2_gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

plt.imshow(img1_gray)
plt.imshow(img2_gray)

orb = cv2.xfeatures2d.SIFT_create(nfeatures=2000)

# Find the key points and descriptors with ORB
keypoints1, descriptors1 = orb.detectAndCompute(img1, None)
keypoints2, descriptors2 = orb.detectAndCompute(img2, None)

plt.imshow(cv2.drawKeypoints(img1, keypoints1, None, (255, 0, 255)))
plt.imshow(cv2.drawKeypoints(img2, keypoints2, None, (255, 0, 255)))
# bf = cv2.BFMatcher_create(cv2.NORM_HAMMING)
bf = cv2.BFMatcher()
# Find matching points
matches = bf.knnMatch(descriptors1, descriptors2,k=2)


def draw_matches(img1, keypoints1, img2, keypoints2, matches):
    r, c = img1.shape[:2]
    r1, c1 = img2.shape[:2]

    # Create a blank image with the size of the first image + second image
    output_img = np.zeros((max([r, r1]), c + c1, 3), dtype='uint8')
    output_img[:r, :c, :] = np.dstack([img1, img1, img1])
    output_img[:r1, c:c + c1, :] = np.dstack([img2, img2, img2])

    # Go over all of the matching points and extract them
    for match in matches:
        img1_idx = match.queryIdx
        img2_idx = match.trainIdx
        (x1, y1) = keypoints1[img1_idx].pt
        (x2, y2) = keypoints2[img2_idx].pt

        # Draw circles on the keypoints
        cv2.circle(output_img, (int(x1), int(y1)), 4, (0, 255, 255), 1)
        cv2.circle(output_img, (int(x2) + c, int(y2)), 4, (0, 255, 255), 1)

        # Connect the same keypoints
        cv2.line(output_img, (int(x1), int(y1)), (int(x2) + c, int(y2)), (0, 255, 255), 1)

    return output_img

all_matches = []
for m, n in matches:
  all_matches.append(m)

img3 = draw_matches(img1_gray, keypoints1, img2_gray, keypoints2, all_matches[:30])
plt.imshow(img3)

good = []
for m, n in matches:
    if m.distance < 0.6 * n.distance:
        good.append(m)

plt.imshow(cv2.drawKeypoints(img1, [keypoints1[m.queryIdx] for m in good], None, (255, 0, 255)))
plt.imshow(cv2.drawKeypoints(img2, [keypoints2[m.trainIdx] for m in good], None, (255, 0, 255)))
def warpImages(img1, img2, H):
    rows1, cols1 = img1.shape[:2]
    rows2, cols2 = img2.shape[:2]

    list_of_points_1 = np.float32([[0, 0], [0, rows1], [cols1, rows1], [cols1, 0]]).reshape(-1, 1, 2)
    temp_points = np.float32([[0, 0], [0, rows2], [cols2, rows2], [cols2, 0]]).reshape(-1, 1, 2)

    # When we have established a homography we need to warp perspective
    # Change field of view
    list_of_points_2 = cv2.perspectiveTransform(temp_points, H)

    list_of_points = np.concatenate((list_of_points_1, list_of_points_2), axis=0)

    [x_min, y_min] = np.int32(list_of_points.min(axis=0).ravel() - 0.5)
    [x_max, y_max] = np.int32(list_of_points.max(axis=0).ravel() + 0.5)

    translation_dist = [-x_min, -y_min]

    H_translation = np.array([[1, 0, translation_dist[0]], [0, 1, translation_dist[1]], [0, 0, 1]])

    output_img = cv2.warpPerspective(img2, H_translation.dot(H), (x_max - x_min, y_max - y_min))
    output_img[translation_dist[1]:rows1 + translation_dist[1], translation_dist[0]:cols1 + translation_dist[0]] = img1

    return output_img


MIN_MATCH_COUNT = 10

if len(good) > MIN_MATCH_COUNT:
    # Convert keypoints to an argument for findHomography
    src_pts = np.float32([keypoints1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    dst_pts = np.float32([keypoints2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

    # Establish a homography
    M, _ = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

    result = warpImages(img2, img1, M)

    # cv2_imshow(result)

    plt.imshow(result)
    plt.title("final image")
    plt.show()
    cv2.imwrite("aut_12.png",result)