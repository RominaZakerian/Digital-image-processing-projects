import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import random
import os, sys
from PIL import Image

############ part A #############################
im = Image.open('internet_censorship_map.png', 'r').convert('RGB')
plt.imshow(im)
plt.show()
width, height = im.size
data=np.empty((width,height,3))
for y in range(0, height): #each pixel has coordinates
    row = ""
    for x in range(0, width):

        RGB = im.getpixel((x,y))
        # print(RGB)
        data[x,y]=RGB

print(data)
print(data.shape)
earth=0
for i in range(data.shape[0]):
    for j in range(data.shape[1]):
        if (not(data[i,j]==[0,0,0]).all()):
            earth=earth+1

print("this is the area of earth",earth)
water=0
for i in range(data.shape[0]):
    for j in range(data.shape[1]):
        if (not(data[i,j]==[0,0,0]).all()) and (not(data[i,j]==[255,153,221]).all()) and (not(data[i,j]==[255,221,221]).all()) \
                and (not (data[i, j] == [255,255,221]).all()) and (not(data[i,j]==[152,251,152]).all()) and (not(data[i,j]==[224,224,224]).all()):
            water=water+1

print("this is the area of water",water)
print("this is the percentage of water:",water/earth)

countries=earth-water
print("this is area of countries:",countries)

pervasive_censorship=0
for i in range(data.shape[0]):
    for j in range(data.shape[1]):
        if ((data[i,j]==[255,153,221]).all()):
            pervasive_censorship=pervasive_censorship+1

print("this is area of countries:",countries)
print("this is percentage of pervasive_censorship:",pervasive_censorship/countries)

c=0
for i in range(data.shape[0]):
    for j in range(data.shape[1]):
        if ( data[i, j] == [255,255,221]).all() or ((data[i,j]==[152,251,152]).all()):
            c=c+1

print("this is percentage of c:",c/countries)

############# part B ############################

im = Image.open('provinces_of_iran_by_population.png', 'r').convert('RGB')
plt.imshow(im)
plt.show()

width, height = im.size
data=np.empty((width,height,3))
for y in range(0, height): #each pixel has coordinates
    for x in range(0, width):

        RGB = im.getpixel((x,y))
        # print(RGB)
        data[x,y]=RGB

print(data)
print(data.shape)

area_of_iran=0
for i in range(data.shape[0]):
    for j in range(data.shape[1]):
        if (not(data[i,j]==[0,0,0]).all()):
            area_of_iran=area_of_iran+1
print("this is the area of iran:",area_of_iran)

d=0
for i in range(data.shape[0]):
    for j in range(data.shape[1]):
        if (data[i,j]==[0,0,51]).all():
            d=d+1
print("this is the percentage of provinces with more than 4 million population:",d/area_of_iran)

e=0
for i in range(data.shape[0]):
    for j in range(data.shape[1]):
        if ((data[i,j]==[0,0,51]).all()) or ((data[i,j]==[0,0,102]).all()) or ((data[i,j]==[51,51,153]).all()):
            e=e+1
print("this is the percentage of provinces with more than 2 million population:",e/area_of_iran)

# plt.show()


f=0
for i in range(328,data.shape[0]):
    for j in range(data.shape[1]):
        if ((data[i,j]==[204,204,255]).all()):
            f=f+1
print("this is the percentage of provinces with more than 1 million population in the eastern part of the country:",f/area_of_iran)

color1=0
color2=0
color3=0
color4=0
color5=0
color6=0
g1=0
for i in range(data.shape[0]):
    for j in range(int(data.shape[1]/2)):
        if (not(data[i,j]==[0,0,0]).all() and (data[i,j]==[204,204,255]).all() ):
            color1=color1+1
        elif (not(data[i,j]==[0,0,0]).all() and (data[i,j]==[153,153,204]).all() ):
            color2=color2+1
        elif (not(data[i,j]==[0,0,0]).all() and (data[i,j]==[102,102,204]).all() ):
            color3=color3+1
        elif ((not(data[i,j]==[0,0,0]).all()) and (data[i,j]==[51,51,153]).all() ):
            color4=color4+1
        elif ((not(data[i,j]==[0,0,0]).all()) and (data[i,j]==[0,0,102]).all() ):
            color5=color5+1
        elif ((not(data[i,j]==[0,0,0]).all()) and (data[i,j]==[0,0,51]).all() ):
            color6=color6+1
#
# pop_col1=0
# pop_col2=0
# pop_col3=0
# pop_col4=0
# pop_col5=0
# pop_col6=0

pop_col1=color1*1
pop_col2=color2*1.5
pop_col3=color3*2
pop_col4=color4*3
pop_col5=color5*4
pop_col6=color6*4.1

population_in_north=pop_col1+pop_col2+pop_col3+pop_col4+pop_col5+pop_col6
print("this is the approximation of population in northern provinces:",population_in_north)

gcolor1=0
color2=0
color3=0
color4=0
color5=0
color6=0
# g1=0
for i in range(data.shape[0]):
    for j in range(int(data.shape[1]/2),int(data.shape[1])):
        if ((not(data[i,j]==[0,0,0]).all()) and (data[i,j]==[204,204,255]).all() ):
            color1=color1+1
        elif ((not(data[i,j]==[0,0,0]).all()) and (data[i,j]==[153,153,204]).all()  ):
            color2=color2+1
        elif ((not(data[i,j]==[0,0,0]).all()) and (data[i,j]==[102,102,204]).all()  ):
            color3=color3+1
        elif ((not(data[i,j]==[0,0,0]).all()) and (data[i,j]==[51,51,153]).all()  ):
            color4=color4+1
        elif ((not(data[i,j]==[0,0,0]).all()) and (data[i,j]==[0,0,102]).all()  ):
            color5=color5+1
        elif ((not(data[i,j]==[0,0,0]).all()) and (data[i,j]==[0,0,51]).all()  ):
            color6=color6+1
#
# pop_col1=0
# pop_col2=0
# pop_col3=0
# pop_col4=0
# pop_col5=0
# pop_col6=0

pop_col1=color1*1
pop_col2=color2*1.5
pop_col3=color3*2
pop_col4=color4*3
pop_col5=color5*4
pop_col6=color6*4.1

population_in_south=pop_col1+pop_col2+pop_col3+pop_col4+pop_col5+pop_col6
print("this is the approximation of population in southern provinces:",population_in_south)

plt.show()

############## part C ##########################

im = Image.open('iran_population-urban_vs_rural.png', 'r').convert('RGB')
plt.imshow(im)

width, height = im.size
data=np.empty((width,height,3))
R=[]
G=[]
B=[]
for y in range(0, height): #each pixel has coordinates
    for x in range(0, width):

        RGB = im.getpixel((x,y))
        # print(RGB)
        data[x,y]=RGB
        R.append(RGB[0])
        G.append(RGB[1])
        B.append(RGB[2])


print(data)
print(data.shape)

##########  h ###############
red=0
for i in range(data.shape[0]):
    for j in range(data.shape[1]):
        if (data[i,j,0]==255 and data[i,j,1]!=255  and data[i,j,2]!=255 ):
            red=red+1
#
green=0
for i in range(data.shape[0]):
    for j in range(data.shape[1]):
        if ((data[i,j]==[51,153,51]).all()):
            green=green+1

print("this is the area of population:",red+green)
print("this is the ratio between the number of urban residents to the total population of the country over the given period.:",red/(red+green))

###########  i ############
green=0
for i in range(602,722):
    for j in range(data.shape[1]):
        if ((data[i,j]==[51,153,51]).all()):
            green=green+1
red=0
for i in range(602,722):
    for j in range(data.shape[1]):
        if ((data[i,j]==[255,51,51]).all()):
            red=red+1

print("this is the area between 1996 and 2006:",red+green)
print("this is percentage of rural residents between the years 1996 to 2006",green/(red+green))

########### part j ############
green=0
for i in range(369,data.shape[0]):
    for j in range(data.shape[1]):
        if ((data[i,j]==[51,153,51]).all()):
            green=green+1

print("this is the exact number of city dwellers in the decade started from the year 1976:",green)
# print("this is percentage of rural residents between the years 1996 to 2006",green/(red+green))


###########  k ############
green=0
red=0
temp=[]
for i in range(data.shape[0]):
    for j in range(data.shape[1]):
        if ((data[i,j]==[51,153,51]).all()):
            green=green+1
        elif ((data[i,j]==[255,51,51]).all()):
            red=red+1
    if red>green:
        temp.append(i)
        # print("this is the first position",j)
        # break
    # break
print("this is the first position",temp[0])
# print("this is the exact number of city dwellers in the decade started from the year 1976:",green)
# print("this is percentage of rural residents between the years 1996 to 2006",green/(red+green))

plt.show()

############# part D #########################


im = Image.open('iran_population_pyramid.png', 'r').convert('RGB')
plt.imshow(im)

width, height = im.size
data=np.empty((width,height,3))
R=[]
G=[]
B=[]
for y in range(0, height): #each pixel has coordinates
    for x in range(0, width):

        RGB = im.getpixel((x,y))
        # print(RGB)
        data[x,y]=RGB
        R.append(RGB[0])
        G.append(RGB[1])
        B.append(RGB[2])


print(data)
print(data.shape)
# plt.show()

######## L ###########
women=0
men=0
for i in range(data.shape[0]):
    for j in range(data.shape[1]):
        if ((data[i,j]==[0,176,240]).all()):
            women=women+1
        elif ((data[i,j]==[255,0,0]).all()):
            men=men+1

print("women's population:",women)
print("men's population:",men)


######### M ###########
genders=women+men
print("this is the percentage of women:",women/genders)
print("this is the percentage of men:",men/genders)

####### N ############
women=0
for i in range(data.shape[0]):
    for j in range(1350,data.shape[1]):
        if ((data[i,j]==[0,176,240]).all()):
            women=women+1

men=0
for i in range(data.shape[0]):
    for j in range(1342,data.shape[1]):
        if ((data[i,j]==[255,0,0]).all()):
            men=men+1

print("population of female under 20:",women)
print("population of male under 20:",men)

######### O #########

men=0
for i in range(data.shape[0]):
    for j in range(852,1032):
        if ((data[i,j]==[255,0,0]).all()):
            men=men+1

print("the exact number of men in their 40’s:",men)
plt.show()
