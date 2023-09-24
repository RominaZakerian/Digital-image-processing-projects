import matplotlib.pyplot as plt
import numpy as np
import matplotlib.image as mpimg
import random
from colormap import rgb2hex

img = mpimg.imread('the_dress.png')
data=np.asarray(img)
imgplot = plt.imshow(img)
print(data)
print(data.shape)
# plt.figure()
# plt.show()

k_mean=5
#
centroids_index_i=random.sample(range(750), k_mean)
centroids_index_j=random.sample(range(750), k_mean)

centroid=np.empty((k_mean,4))
new_centroid=np.empty((k_mean,4))

i=0
j=0
print(centroids_index_i)
print(centroids_index_j)
while(i<len(centroids_index_i) and j<len(centroids_index_j)):
    centroid[i,:]=data[centroids_index_i[i],centroids_index_j[j],:]
    i=i+1
    j=j+1
print("this is centroids\n",centroid)

def distance_of_2_pixels(x,y):
    return np.sqrt(sum(pow((x-y),2)))

# print(distance_of_2_pixels(centroid[0],centroid[1]))
#
cluster_number=np.zeros((k_mean,))
cluster=np.zeros(((k_mean,4)))
print(cluster_number.shape)
for i in range(data.shape[0]):
    for j in range(data.shape[1]):
        distance=[]
        x=data[i,j,:]
        for k in range(centroid.shape[0]):
            dist=distance_of_2_pixels(centroid[k,:],x)
            distance.append(dist)
        # print(distance)
        m = distance.index(min(distance))
        # print(m)
        cluster_number[m]=cluster_number[m]+1
        # print(cluster_number)
        cluster[m]=cluster[m]+data[i,j]
        # print(cluster)
        # data[i,j]=centroid[m]
# print(data)
print("########## the first cluster ############")
print("this is cluster number\n",cluster_number)
print("this is sum of clusters:\n",cluster)
k=3
i=0
for i in range(k_mean):
    print(i,cluster[i],cluster_number[i])

    new_centroid[i]=cluster[i]/cluster_number[i]
    print(new_centroid[i])
print("this is previous centroids\n",centroid)
print(" this is new centroid:\n",new_centroid)

while(not(np.array_equal(centroid,new_centroid))):
    k=3
    cluster_number = np.zeros((k_mean,))
    print(cluster_number.shape)
    cluster = np.zeros(((k_mean, 4)))
    centroid=new_centroid
    print("this is centroid for continue",centroid)
    new_centroid = np.empty((k_mean, 4))
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            distance = []
            x = data[i, j, :]
            for k in range(centroid.shape[0]):
                dist = distance_of_2_pixels(centroid[k, :], x)
                distance.append(dist)
            m = distance.index(min(distance))
            cluster_number[m] = cluster_number[m] + 1
            cluster[m] = cluster[m] + data[i, j]
            # print(m)
            # data[i,j]=centroid[m]
    # print(data)
    print("############################")
    print("this is cluster number\n",cluster_number)
    print("this is sum of clusters\n",cluster)

    for i in range(k_mean):
        new_centroid[i] = cluster[i] / cluster_number[i]
    print("this is previous centroid\n",centroid)
    print(" this is new centroid\n", new_centroid)
    print("############################")

print("done")
# centroid=np.empty((k_mean,4))



for i in range(data.shape[0]):
    for j in range(data.shape[1]):
        distance = []
        x = data[i, j, :]
        for k in range(centroid.shape[0]):
            dist = distance_of_2_pixels(centroid[k, :], x)
            distance.append(dist)
        m = distance.index(min(distance))
        cluster_number[m] = cluster_number[m] + 1
        # cluster[m] = cluster[m] + data[i, j]
        # print(m)
        data[i,j]=centroid[m,:]

print(data)
# data=data.reshape(-1,1)
print(data.shape)
plt.imshow(data)
plt.show()

colors=[centroid[j] for j in range(k_mean)]
fig1, ax1 = plt.subplots()
ax1.pie(cluster_number, colors=colors, autopct='%1.1f%%',
        shadow=True, startangle=90)
ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.

plt.show()