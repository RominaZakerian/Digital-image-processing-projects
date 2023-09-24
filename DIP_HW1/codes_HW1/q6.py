import cv2
import matplotlib.pyplot as plt
import numpy as np

im = cv2.imread('tetris.png')
print(im)
print(im.shape)
plt.figure()
plt.imshow(cv2.cvtColor(im, cv2.COLOR_BGR2RGB))
plt.show()
im1=np.empty((im.shape[0],im.shape[1]))
for i in range(45,im.shape[0],100):
    for j in range(45,im.shape[1],100):
         print(im[i,j])
         if (im[i,j]==[0,0,0]).all():
             im1[i,j]=0
         elif (im[i,j]==[204,0,204]).all():
             im1[i, j] = 7
         elif (im[i,j]==[0,255,255]).all():
             im1[i, j] = 3
         elif (im[i,j]==[255,255,0]).all():
             im1[i, j] = 2
         elif (im[i,j]==[0,255,0]).all():
             im1[i, j] = 5
         elif (im[i,j]==[0,0,255]).all():
             im1[i, j] = 8
         elif (im[i,j]==[0,119,255]).all():
             im1[i, j] = 4
         elif (im[i,j]==[170,0,0]).all():
             im1[i, j] = 6
         elif (im[i,j]==[119,119,119]).all():
             im1[i, j] = 1
         # print(i,j)

print(im1)
print(im1.shape)


boolean=True
while(boolean):
    value = input("Enter the shape of your piece(T,S,O,I,L): ")
    print(value)
    if value=="T":
        cordinate_x=45
        # cordinate_y=645
        val = input("Enter your value that is the multiplication of 100 and sum it with 45 between (145,1045): ")
        print(val)
        cordinate_y=int(val)
        val = input("Enter the rotation of T(1(0),2(45),3(90),4(135): ")
        print(val)
        ############# play T ##############
        while( cordinate_x<im.shape[0]):
            i = cordinate_x
            j = cordinate_y
            if val=="1":
                if j>145 and j<1045 and im1[i+100,j]==0 and im1[i+100,j+100]==0 and im1[i+100,j-100]==0 and im1[i+200,j]==0:
                    print(i)
                    for k in range(i+45,i+145):
                        for l in range(j-145,j+155):
                            print(k,l)
                            im[k,l]=[204,0,204]
                    for k in range(i+145,i+245):
                        for l in range(j-45,j+45):
                            im[k, l] = [204, 0, 204]
                elif j==145 and im1[i+100,j]==0 and im1[i+100,j+100]==0 and im1[i+100,j+200]==0 and im1[i+200,j+100]==0:
                    for k in range(i+45,i+145):
                        for l in range(j-145,j+245):
                            print(k,l)
                            im[k,l]=[204,0,204]
                    for k in range(i+145,i+245):
                        for l in range(j+45,j+145):
                            im[k, l] = [204, 0, 204]
                elif j == 1045 and im1[i + 100, j] == 0 and im1[i + 100, j - 100] == 0 and im1[i + 100, j - 200] == 0 and im1[i + 200, j - 100] == 0:
                    for k in range(i + 45, i + 145):
                        for l in range(j - 245, j + 45):
                            print(k, l)
                            im[k, l] = [204, 0, 204]
                    for k in range(i + 145, i + 245):
                        for l in range(j - 145, j - 45):
                            im[k, l] = [204, 0, 204]
                else:
                    break
            elif val=="2":
                if j>=145 and j<1045 and im1[i+100,j]==0 and im1[i+200,j]==0 and im1[i+300,j-100]==0 and im1[i+200,j+100]==0:
                    print(i)
                    for k in range(i+45,i+345):
                        for l in range(j-45,j+45):
                            print(k,l)
                            im[k,l]=[204,0,204]
                    for k in range(i+145,i+245):
                        for l in range(j+45,j+145):
                            im[k, l] = [204, 0, 204]
                elif j==1045 and im1[i+200,j]==0 and im1[i+200,j-100]==0 and im1[i+300,j-100]==0 and im1[i+100,j-100]==0:

                    for k in range(i+145,i+245):
                        for l in range(j-45,j+45):
                            print(k,l)
                            im[k,l]=[204,0,204]
                    for k in range(i+45,i+345):
                        for l in range(j-145,j-45):
                            im[k, l] = [204, 0, 204]
                else:
                    break
            elif val == "3":
                if j > 145 and j < 1045 and im1[i + 200, j] == 0 and im1[i + 200, j-100] == 0 and im1[i + 200, j + 100] == 0 and \
                        im1[i + 100, j ] == 0:
                    print(i)
                    for k in range(i + 45, i + 145):
                        for l in range(j - 45, j + 45):
                            print(k, l)
                            im[k, l] = [204, 0, 204]
                    for k in range(i + 145, i + 245):
                        for l in range(j - 145, j + 145):
                            im[k, l] = [204, 0, 204]
                elif j == 1045 and im1[i + 200, j] == 0 and im1[i + 200, j - 100] == 0 and im1[i + 300, j - 200] == 0 and im1[i + 100, j ] == 0:

                    for k in range(i + 45, i + 145):
                        for l in range(j - 145, j - 45):
                            print(k, l)
                            im[k, l] = [204, 0, 204]
                    for k in range(i + 145, i + 245):
                        for l in range(j - 245, j + 45):
                            im[k, l] = [204, 0, 204]
                elif j == 145 and im1[i + 200, j] == 0 and im1[i + 200, j + 100] == 0 and im1[i + 200, j + 200] == 0 and im1[i + 100, j+100 ] == 0:

                    for k in range(i + 45, i + 145):
                        for l in range(j + 45, j + 145):
                            print(k, l)
                            im[k, l] = [204, 0, 204]
                    for k in range(i + 145, i + 245):
                        for l in range(j - 45, j + 245):
                            im[k, l] = [204, 0, 204]
                else:
                    break
            elif val == "4":
                if j > 145 and j <= 1045 and im1[i + 100, j] == 0 and im1[i + 200, j] == 0 and im1[i + 300, j] == 0 and \
                        im1[i + 200, j - 100] == 0:
                    print(i)
                    for k in range(i + 45, i + 345):
                        for l in range(j - 45, j + 45):
                            print(k, l)
                            im[k, l] = [204, 0, 204]
                    for k in range(i + 145, i + 245):
                        for l in range(j - 145, j - 45):
                            im[k, l] = [204, 0, 204]
                elif j == 145 and im1[i + 200, j] == 0 and im1[i + 100, j + 100] == 0 and im1[i + 200, j + 100] == 0 and im1[i + 200, j + 100] == 0:

                    for k in range(i + 145, i + 245):
                        for l in range(j - 45, j + 45):
                            print(k, l)
                            im[k, l] = [204, 0, 204]
                    for k in range(i + 45, i + 345):
                        for l in range(j + 45, j + 145):
                            im[k, l] = [204, 0, 204]
                else:
                    break


            plt.imshow(cv2.cvtColor(im.astype(np.uint8), cv2.COLOR_BGR2RGB))
            plt.show()
            cordinate_x=cordinate_x+100
            if val=="1":
                if j>145 and j<1045 :
                    print(i)
                    for k in range(i+45,i+145):
                        for l in range(j-145,j+155):
                            print(k,l)
                            im[k,l]=[0,0,0]
                    for k in range(i+145,i+245):
                        for l in range(j-45,j+45):
                            im[k, l] = [0, 0, 0]
                elif j == 145 :
                    for k in range(i + 45, i + 145):
                        for l in range(j - 145, j + 245):
                            print(k, l)
                            im[k, l] = [0, 0, 0]
                    for k in range(i + 145, i + 245):
                        for l in range(j + 45, j + 145):
                                    im[k, l] = [0, 0, 0]
                elif j == 1045 :
                    for k in range(i + 45, i + 145):
                        for l in range(j - 245, j + 45):
                            print(k, l)
                            im[k, l] = [0, 0, 0]
                    for k in range(i + 145, i + 245):
                        for l in range(j - 145, j - 45):
                            im[k, l] = [0, 0, 0]
                else:
                    break
            elif val=="2":
                if j>=145 and j<1045 :
                    print(i)
                    for k in range(i+45,i+345):
                        for l in range(j-45,j+45):
                            print(k,l)
                            im[k,l]=[0,0,0]
                    for k in range(i+145,i+245):
                        for l in range(j+45,j+145):
                            im[k, l] = [0, 0, 0]
                elif j==1045 :

                    for k in range(i+145,i+245):
                        for l in range(j-45,j+45):
                            print(k,l)
                            im[k,l]=[0,0,0]
                    for k in range(i+45,i+345):
                        for l in range(j-145,j-45):
                            im[k, l] = [0, 0, 0]
                else:
                    break
            elif val == "3":
                if j > 145 and j < 1045 :
                    print(i)
                    for k in range(i + 45, i + 145):
                        for l in range(j - 45, j + 45):
                            print(k, l)
                            im[k, l] = [0, 0, 0]
                    for k in range(i + 145, i + 245):
                        for l in range(j - 145, j + 145):
                            im[k, l] = [0, 0, 0]
                elif j == 1045 :

                    for k in range(i + 45, i + 145):
                        for l in range(j - 145, j - 45):
                            print(k, l)
                            im[k, l] = [0, 0, 0]
                    for k in range(i + 145, i + 245):
                        for l in range(j - 245, j + 45):
                            im[k, l] = [0, 0, 0]
                elif j == 145 :

                    for k in range(i + 45, i + 145):
                        for l in range(j + 45, j + 145):
                            print(k, l)
                            im[k, l] = [0, 0, 0]
                    for k in range(i + 145, i + 245):
                        for l in range(j - 45, j + 245):
                            im[k, l] = [0, 0, 0]
                else:
                    break
            elif val == "4":
                if j > 145 :
                    print(i)
                    for k in range(i + 45, i + 345):
                        for l in range(j - 45, j + 45):
                            print(k, l)
                            im[k, l] = [0, 0, 0]
                    for k in range(i + 145, i + 245):
                        for l in range(j - 145, j - 45):
                            im[k, l] = [0, 0, 0]
                elif j == 145 :

                    for k in range(i + 145, i + 245):
                        for l in range(j - 45, j + 45):
                            print(k, l)
                            im[k, l] = [0, 0, 0]
                    for k in range(i + 45, i + 345):
                        for l in range(j + 45, j + 145):
                            im[k, l] = [0, 0, 0]
                else:
                    break
    elif value=="L":
        ############# play L ##############

        cordinate_x=45
        # cordinate_y=645

        val = input("Enter your value that is the multiplication of 100 and sum it with 45 between (45,1045): ")
        print(val)
        cordinate_y=int(val)
        val = input("Enter the rotation of S(1(0),2(45),3(90),4(135): ")
        print(val)
        while( cordinate_x<im.shape[0]):
            i = cordinate_x
            j = cordinate_y
            if val=="1":
                if j>=145 and j!=1045 and im1[i+100,j]==0 and im1[i+200,j]==0 and im1[i+300,j]==0 and im1[i+300,j+100]==0:
                    print(i)
                    for k in range(i+45,i+345):
                        for l in range(j-45,j+45):
                            print(k,l)
                            im[k,l]=[170,0,0]
                    for k in range(i+245,i+345):
                        for l in range(j+45,j+145):
                            im[k, l] = [170, 0, 0]
                elif j==1045:
                    for k in range(i+45,i+345):
                        for l in range(j-45,j+45):
                            print(k,l)
                            im[k,l]=[170,0,0]
                    for k in range(i+45,i+145):
                        for l in range(j-145,j-45):
                            im[k, l] = [170, 0, 0]
                else:
                    break
            elif val=="2":
                if j>=145 and j<=845 and im1[i+100,j]==0 and im1[i+200,j]==0 and im1[i+100,j+100]==0 and im1[i+100,j+200]==0:
                    print(i)
                    for k in range(i+45,i+245):
                        for l in range(j-45,j+45):
                            print(k,l)
                            im[k,l]=[170,0,0]
                    for k in range(i+45,i+145):
                        for l in range(j-45,j+245):
                            im[k, l] = [170, 0, 0]
                elif j>845:
                    val="1"
                    cordinate_x=i
                    continue
                else:
                    break
            elif val=="3":
                if j>=145 and j<=945 and im1[i+100,j]==0 and im1[i+100,j+100]==0 and im1[i+200,j+100]==0 and im1[i+300,j+100]==0:
                    print(i)
                    for k in range(i+45,i+145):
                        for l in range(j-45,j+145):
                            print(k,l)
                            im[k,l]=[170,0,0]
                    for k in range(i+145,i+345):
                        for l in range(j+45,j+145):
                            im[k, l] = [170, 0, 0]
                elif j>945 and im1[i+100,j]==0 and im1[i+200,j]==0 and im1[i+300,j]==0 and im1[i+100,j-100]==0:

                    for k in range(i+45,i+145):
                        for l in range(j-145,j+45):
                            print(k,l)
                            im[k,l]=[170,0,0]
                    for k in range(i+145,i+345):
                        for l in range(j-45,j+45):
                            im[k, l] = [170, 0, 0]
                            im[k, l] = [170, 0, 0]
                else:
                    break
            elif val == "4":
                if j >= 145 and j <= 845 and im1[i + 200, j] == 0 and im1[i + 200, j + 100] == 0 and im1[i + 200, j + 200] == 0 and im1[i + 100, j + 200] == 0:
                    print(i,"hello")
                    for k in range(i + 145, i + 245):
                        for l in range(j - 45, j + 245):
                            print(k, l)
                            im[k, l] = [170, 0, 0]
                    for k in range(i + 45, i + 145):
                        for l in range(j + 145, j + 245):
                            im[k, l] = [170, 0, 0]
                elif j > 845 and ((im1[i + 200, j] == 0 and im1[i + 200, j - 100] == 0 and im1[i + 200, j - 200] == 0 and im1[i + 100, j]==0)):

                    for k in range(i + 45, i + 245):
                        for l in range(j - 45, j + 45):
                            print(k, l)
                            im[k, l] = [170, 0, 0]
                    for k in range(i + 145, i + 245):
                        for l in range(j - 245, j - 45):
                            im[k, l] = [170, 0, 0]
                            im[k, l] = [170, 0, 0]
                else:
                    break
            plt.imshow(cv2.cvtColor(im.astype(np.uint8), cv2.COLOR_BGR2RGB))
            plt.show()
            cordinate_x=cordinate_x+100
            if val=="1":
                if j>=145 and j!=1045 :
                    print(i)
                    for k in range(i+45,i+345):
                        for l in range(j-45,j+45):
                            print(k,l)
                            im[k,l]=[0,0,0]
                    for k in range(i+245,i+345):
                        for l in range(j+45,j+145):
                            im[k, l] = [0, 0, 0]
                elif j==1045:
                    for k in range(i+45,i+345):
                        for l in range(j-45,j+45):
                            print(k,l)
                            im[k,l]=[0,0,0]
                    for k in range(i+45,i+145):
                        for l in range(j-145,j-45):
                            im[k, l] = [0, 0, 0]

            elif val=="2":
                if j>=145 and j<=845 :
                    print(i)
                    for k in range(i+45,i+245):
                        for l in range(j-45,j+45):
                            print(k,l)
                            im[k,l]=[0,0,0]
                    for k in range(i+45,i+145):
                        for l in range(j-45,j+245):
                            im[k, l] = [0, 0, 0]
                elif j>845:
                    if j >= 145 and j != 1045:
                        print(i)
                        for k in range(i + 45, i + 345):
                            for l in range(j - 45, j + 45):
                                print(k, l)
                                im[k, l] = [0, 0, 0]
                        for k in range(i + 245, i + 345):
                            for l in range(j + 45, j + 145):
                                im[k, l] = [0, 0, 0]
                    elif j == 1045:
                        for k in range(i + 45, i + 345):
                            for l in range(j - 45, j + 45):
                                print(k, l)
                                im[k, l] = [0, 0, 0]
                        for k in range(i + 45, i + 145):
                            for l in range(j - 145, j - 45):
                                im[k, l] = [0, 0, 0]

            elif val=="3":
                if j>=145 and j<=945 :
                    print(i)
                    for k in range(i+45,i+145):
                        for l in range(j-45,j+145):
                            print(k,l)
                            im[k,l]=[0,0,0]
                    for k in range(i+145,i+345):
                        for l in range(j+45,j+145):
                            im[k, l] = [0, 0, 0]
                elif j>945:

                    for k in range(i+45,i+145):
                        for l in range(j-145,j+45):
                            print(k,l)
                            im[k,l]=[0,0,0]
                    for k in range(i+145,i+345):
                        for l in range(j-45,j+45):
                            im[k, l] = [0, 0, 0]
                            im[k, l] = [0, 0, 0]

            elif val == "4":
                if j >= 145 and j <= 845 :
                    print(i)
                    for k in range(i + 145, i + 245):
                        for l in range(j - 45, j + 245):
                            print(k, l)
                            im[k, l] = [0, 0, 0]
                    for k in range(i + 45, i + 145):
                        for l in range(j + 145, j + 245):
                            im[k, l] = [0, 0, 0]
                elif j > 845:

                    for k in range(i + 45, i + 245):
                        for l in range(j - 45, j + 45):
                            print(k, l)
                            im[k, l] = [0, 0, 0]
                    for k in range(i + 145, i + 245):
                        for l in range(j - 245, j - 45):
                            im[k, l] = [0, 0, 0]

    elif value=="I":

        ############# play I ##############

        cordinate_x=45
        # cordinate_y=645
        val = input("Enter your value that is the multiplication of 100 and sum it with 45 between (45,1045): ")
        print(val)
        cordinate_y=int(val)
        val = input("Enter the rotation of I(H,V): ")
        print(val)
        #
        while( cordinate_x<im.shape[0]):
            i = cordinate_x
            j = cordinate_y
            if val=="V":

                if im1[i+100,j]==0 and im1[i+200,j]==0 and im1[i+300,j]==0 and im1[i+400,j]==0:
                    print(i)
                    for k in range(i+45,i+445):
                        for l in range(j-45,j+45):
                            print(k,l)
                            im[k,l]=[255,255,0]
                    # for k in range(i+245,i+345):
                    #     for l in range(j+45,j+145):
                    #         im[k, l] = [170, 0, 0]
                else:
                    break
            else: # if val="H"
                if j+300<1045 and im1[i+100,j]==0 and im1[i+100,j+100]==0 and im1[i+100,j+200]==0 and im1[i+100,j+300]==0:

                    for k in range(i+45,i+145):
                        for l in range(j-45,j+345):
                            print(k,l)
                            im[k,l]=[255,255,0]
                elif  j+300>1045 and im1[i+100,j]==0 and im1[i+100,j-100]==0 and im1[i+100,j-200]==0 and im1[i+100,j-300]==0:
                    for k in range(i+45,i+145):
                        for l in range(j-345,j+45):
                            print(k,l)
                            im[k,l]=[255,255,0]
                else:
                    break
            plt.imshow(cv2.cvtColor(im.astype(np.uint8), cv2.COLOR_BGR2RGB))
            plt.show()
            cordinate_x=cordinate_x+100

            if val=="V":
                for k in range(i + 45, i + 445):
                    for l in range(j - 45, j + 45):
                        print(k, l)
                        im[k, l] = [0, 0, 0]
            elif val=="H":
                if j+300<1045:
                    for k in range(i + 45, i + 145):
                        for l in range(j - 45, j + 345):
                            print(k, l)
                            im[k, l] = [0, 0, 0]
                elif j + 300 > 1045:
                    for k in range(i + 45, i + 145):
                        for l in range(j - 345, j + 45):
                            print(k, l)
                            im[k, l] = [0, 0, 0]
    elif value=="O":
        ############ play O ##############

        cordinate_x=45
        val = input("Enter your value that is the multiplication of 100 and sum it with 45 between (145,1045): ")
        print(val)
        cordinate_y=int(val)


        while( cordinate_x<im.shape[0]):
            i = cordinate_x
            j = cordinate_y
            if cordinate_y==1045:
                if im1[i + 100, j] == 0 and im1[i + 200, j] == 0 and im1[i + 100, j - 100] == 0 and im1[i + 200, j - 100] == 0:
                    for k in range(i + 45, i + 245):
                        for l in range(j - 145, j + 45):
                            print(k, l)
                            im[k, l] = [0, 255, 255]
                else:
                    break
            else:

                if im1[i+100,j]==0 and im1[i+200,j]==0 and im1[i+100,j+100]==0 and im1[i+200,j+100]==0:
                    print(i)
                    for k in range(i+45,i+245):
                        for l in range(j-45,j+145):
                            print(k,l)
                            im[k,l]=[0,255,255]
                    # for k in range(i+245,i+345):
                    #     for l in range(j+45,j+145):
                    #         im[k, l] = [170, 0, 0]
                else:
                    break
            plt.imshow(cv2.cvtColor(im.astype(np.uint8), cv2.COLOR_BGR2RGB))
            plt.show()
            cordinate_x=cordinate_x+100
            if cordinate_y!=1045:
                for k in range(i + 45, i + 245):
                    for l in range(j - 45, j + 145):
                        print(k, l)
                        im[k, l] = [0, 0, 0]
            else:
                for k in range(i + 45, i + 245):
                    for l in range(j - 145, j + 45):
                        print(k, l)
                        im[k, l] = [0, 0, 0]

    elif value=="S":
        ############ play S ##############

        cordinate_x=45
        # cordinate_y=645
        val = input("Enter your value that is the multiplication of 100 and sum it with 45 between (145,1045): ")
        print(val)
        cordinate_y=int(val)
        val = input("Enter the rotation of S(H,V): ")
        print(val)

        while( cordinate_x<im.shape[0]):
            i = cordinate_x
            j = cordinate_y
            if val=="V":
                if j+100<1045 and  im1[i+100,j]==0 and im1[i+200,j]==0 and im1[i+200,j+100]==0 and im1[i+300,j+100]==0:
                    print(i)
                    for k in range(i+45,i+245):
                        for l in range(j-45,j+45):
                            print(k,l)
                            im[k,l]=[0,255,0]
                    for k in range(i+145,i+345):
                        for l in range(j+45,j+145):
                            im[k, l] = [0, 255, 0]
                elif im1[i+100,j]==0 and im1[i+200,j]==0 and im1[i+200,j-100]==0 and im1[i+300,j-100]==0:
                    for k in range(i+45,i+245):
                        for l in range(j-45,j+45):
                            print(k,l)
                            im[k,l]=[0,255,0]
                    for k in range(i+145,i+345):
                        for l in range(j-145,j-45):
                            im[k, l] = [0, 255, 0]

                else:
                    break
            elif val=="H": # val="H"
                if j>145 and j!=1045 and im1[i+100,j]==0 and im1[i+200,j]==0 and im1[i+200,j-100]==0 and im1[i+100,j+100]==0:
                    print(i)
                    print("hello1")
                    for k in range(i+45,i+245):
                        for l in range(j-45,j+45):
                            print(k,l)
                            im[k,l]=[0,255,0]

                    for k in range(i+45,i+145):
                        for l in range(j+45,j+145):
                            print(k,l)
                            im[k,l]=[0,255,0]
                    for k in range(i+145,i+245):
                        for l in range(j-145,j-45):
                            print(k,l)
                            im[k,l]=[0,255,0]
                elif j==145 and im1[i+100,j]==0 and im1[i+100,j+100]==0 and im1[i+200,j+100]==0 and im1[i+200,j+200]==0:
                    print(i)
                    for k in range(i + 45, i + 145):
                        for l in range(j - 45, j + 145):
                            print(k, l)
                            im[k, l] = [0, 255, 0]

                    for k in range(i + 145, i + 245):
                        for l in range(j + 45, j + 245):
                            print(k, l)
                            im[k, l] = [0, 255, 0]
                elif j==1045 and im1[i+100,j]==0 and im1[i+100,j-100]==0 and im1[i+200,j]==0 and im1[i+200,j-200]==0:
                    print(i)
                    print("hello")
                    for k in range(i + 45, i + 145):
                        for l in range(j - 145, j + 45):
                            print(k, l)
                            im[k, l] = [0, 255, 0]

                    for k in range(i + 145, i + 245):
                        for l in range(j - 245, j - 45):
                            print(k, l)
                            im[k, l] = [0, 255, 0]
                else:
                    break
            plt.imshow(cv2.cvtColor(im.astype(np.uint8), cv2.COLOR_BGR2RGB))
            plt.show()
            cordinate_x=cordinate_x+100
            if val=="V":
                if j+100<1045 and  im1[i+100,j]==0 and im1[i+200,j]==0 and im1[i+200,j+100]==0 and im1[i+300,j+100]==0:
                    print(i)
                    for k in range(i+45,i+245):
                        for l in range(j-45,j+45):
                            print(k,l)
                            im[k,l]=[0,0,0]
                    for k in range(i+145,i+345):
                        for l in range(j+45,j+145):
                            im[k, l] = [0,0, 0]
                elif j==1045:
                    for k in range(i+45,i+245):
                        for l in range(j-45,j+45):
                            print(k,l)
                            im[k,l]=[0,0,0]
                    for k in range(i+145,i+345):
                        for l in range(j-145,j-45):
                            im[k, l] = [0, 0, 0]

            elif j>145 and j!=1045:
                for k in range(i + 45, i + 245):
                    for l in range(j - 45, j + 45):
                        print(k, l)
                        im[k, l] = [0, 0, 0]
            elif j==1045:
                for k in range(i + 45, i + 145):
                    for l in range(j - 145, j + 45):
                        print(k, l)
                        im[k, l] = [0, 0, 0]

                for k in range(i + 145, i + 245):
                    for l in range(j - 245, j - 45):
                        print(k, l)
                        im[k, l] = [0, 0, 0]
            elif j==145:
                for k in range(i + 45, i + 145):
                    for l in range(j - 45, j + 145):
                        print(k, l)
                        im[k, l] = [0, 0, 0]

                for k in range(i + 145, i + 245):
                    for l in range(j + 45, j + 245):
                        print(k, l)
                        im[k, l] = [0, 0, 0]

a_file = open("test6.txt", "w")
for row in im1:
    np.savetxt(a_file, row)

a_file.close()

im2 = cv2.imread('I.png')
