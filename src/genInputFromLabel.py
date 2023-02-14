import numpy as np
import matplotlib.pyplot as plt
import cv2
from TreeLib import Tree, drawTree, genTree
import random as rnd

#better idea; segment image. Make a mask. Then sample from either;
#   1)inside mask for artery color.
#   2) outside mask for background color

dim = 736
#path to local data file
pathDir = "../../../Orig_data/"

artery_im1 = pathDir + r"ImsegmentedPt_02 V_0/Orig/IMG00000_30.tiff"

aim1 = cv2.imread(artery_im1, -1)

#Add way to compute mean and std of arteries and background
artery_mean = 70
artery_std = rnd.gauss(artery_mean, 20)
backg_mean = 100
backg_std = rnd.gauss(backg_mean, 20)

#Naive method of generating background and artery colors.
#returns background and artery colors as tuple
def gen_colors():
    background_col = backg_mean + backg_std
    artery_col = artery_mean + artery_std
    #while background is somewhat darker or close to artery color find new
    while (background_col < artery_col + 10):
        background_col = backg_mean + backg_std
        artery_col = artery_mean + artery_std
    #handle under 255 and over 0.
    return int(background_col), int(artery_col)

#compute colors
back_col, art_col = gen_colors()
print("background color: {0}, artery color: {1}".format(back_col, art_col))


#make a tree
tree1 = Tree(0, 362, 15, 15, 0, 2)
#hack to ensure array is int arr not bool arr
synTree = np.array(genTree(tree1, (736, 736))).astype(int)

#insert artery and background color
synTree[synTree > 0] = art_col
synTree[synTree == 0] = back_col
#print(synTree)

#draw input
plt.imshow(synTree, cmap="gray", vmin=0, vmax=255)
plt.show()
#draw label
drawTree(tree1, ((736, 736)))


#Abandoned:

"""
col = np.ones((20, 20)) * backg
col[10] = art
print(col)
plt.imshow(col, cmap = "gray", vmin=0, vmax=255) #vmin, vmax ensures ralistic
plt.show()
"""




"""
back_sample = []

background_im1 = pathDir + r"ImsegmentedPt_02 V_0/Orig/IMG00000_01.tiff"
background_im2 = pathDir + r"ImsegmentedPt_02 V_1/Orig/IMG00001_01.tiff"
background_im3 = pathDir + r"ImsegmentedPt_02 V_3/Orig/IMG00003_01.tiff"
background_im4 = pathDir + r"ImsegmentedPt_02 V_4/Orig/IMG00004_01.tiff"
background_im5 = pathDir + r"ImsegmentedPt_02 V_6/Orig/IMG00006_01.tiff"
background_im6 = pathDir + r"ImsegmentedPt_02 V_7/Orig/IMG00007_01.tiff"
background_im7 = pathDir + r"ImsegmentedPt_02 V_8/Orig/IMG00008_01.tiff"

bim1 = cv2.imread(background_im1, -1)
bim2 = cv2.imread(background_im2, -1)
bim3 = cv2.imread(background_im3, -1)
bim4 = cv2.imread(background_im4, -1)
bim5 = cv2.imread(background_im5, -1)
bim6 = cv2.imread(background_im6, -1)
bim7 = cv2.imread(background_im7, -1)


#print(np.shape(im1))

#plt.imshow(im7, cmap="gray")
#plt.show()

back_sample = genSamples(back_sample, 300, 0, bim1, 100) #for im1: minX=300
back_sample = genSamples(back_sample, 0, 450, bim1, 100) #for im2: minY=450
back_sample = genSamples(back_sample, 300, 0, bim1, 100) #for im3: minX=300
back_sample = genSamples(back_sample, 150, 0, bim1, 100) #for im4: minX=150
back_sample = genSamples(back_sample, 150, 0, bim1, 100) #for im5: minX=150
back_sample = genSamples(back_sample, 250, 0, bim1, 100) #for im6: minX=250
back_sample = genSamples(back_sample, 0, 250, bim1, 100) #for im7: minY=250

back_sample = np.array(back_sample)
mean_background = np.mean(back_sample)
std_background = np.std(back_sample)

print("Mean of background is: {0:.3f} and std is {1:.3f}".format(mean_background, std_background))
#assume background color follows a normal distribution.

#Get artery samples:

artery_im1 = pathDir + r"ImsegmentedPt_02 V_0/Orig/IMG00000_30.tiff"
artery_im2 = pathDir + r"ImsegmentedPt_02 V_1/Orig/IMG00001_30.tiff"
artery_im3 = pathDir + r"ImsegmentedPt_02 V_3/Orig/IMG00003_30.tiff"
artery_im4 = pathDir + r"ImsegmentedPt_02 V_4/Orig/IMG00004_30.tiff"
artery_im5 = pathDir + r"ImsegmentedPt_02 V_6/Orig/IMG00006_30.tiff"
#artery_im6 = pathDir + r"ImsegmentedPt_02 V_7/Orig/IMG00007_30.tiff"
#artery_im7 = pathDir + r"ImsegmentedPt_02 V_8/Orig/IMG00008_30.tiff"

aim1 = cv2.imread(artery_im1, -1)
aim2 = cv2.imread(artery_im2, -1)
aim3 = cv2.imread(artery_im3, -1)
aim4 = cv2.imread(artery_im4, -1)
aim5 = cv2.imread(artery_im5, -1)
#aim6 = cv2.imread(artery_im6, -1)
#aim7 = cv2.imread(artery_im7, -1)

plt.imshow(aim1)
plt.show()

#arteries samples by observing images
a_samples = []

a_samples.append(aim1[305, 21])
a_samples.append(aim1[309, 216])
a_samples.append(aim1[297, 272])
a_samples.append(aim1[220, 272])
a_samples.append(aim1[162, 328])
a_samples.append(aim1[218, 656])
a_samples.append(aim1[581, 590])

a_samples.append(aim2[348, 40])
a_samples.append(aim2[291, 177])
a_samples.append(aim2[366, 305])
a_samples.append(aim2[267, 429])
a_samples.append(aim2[375, 623])
a_samples.append(aim2[353, 651])
a_samples.append(aim2[538, 571])
a_samples.append(aim2[687, 526])

a_samples.append(aim3[105, 89])
a_samples.append(aim3[145, 131])
a_samples.append(aim3[700, 698])
a_samples.append(aim3[491, 530])
a_samples.append(aim3[525, 274])
a_samples.append(aim3[652, 414])
a_samples.append(aim3[422, 226])
a_samples.append(aim3[301, 155])

a_samples.append(aim4[21, 82])
a_samples.append(aim4[611, 574])
a_samples.append(aim4[446, 564])
a_samples.append(aim4[452, 184])
a_samples.append(aim4[81, 441])
a_samples.append(aim4[393, 381])
a_samples.append(aim4[285, 108])
a_samples.append(aim4[581, 239])

a_samples.append(aim5[51, 71])
a_samples.append(aim5[58, 254])
a_samples.append(aim5[676, 656])
a_samples.append(aim5[574, 418])
a_samples.append(aim5[646, 288])
a_samples.append(aim5[404, 289])
a_samples.append(aim5[344, 151])
a_samples.append(aim5[251, 68])

a_samples = np.array(a_samples)
print(a_samples)
print("Mean of artery color: {0:.3f} and std: {1:.3f}".format(np.mean(a_samples), np.std(a_samples)))
"""