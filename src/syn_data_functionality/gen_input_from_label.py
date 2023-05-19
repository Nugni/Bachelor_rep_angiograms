import numpy as np
import matplotlib.pyplot as plt
import random as rnd
from scipy.ndimage import gaussian_filter
from scipy.stats import poisson, lognorm
import sys
sys.path.append("syn_data_functionality")
from syn_data_functionality.bias_field import add_bias_field

# better idea; segment image. Make a mask. Then sample from either;
#   1)inside mask for artery color.
#   2) outside mask for background color

# Add way to compute mean and std of arteries and background
ratio_lognorm_param = [0.09598047931952741, 0.005122247005933689, 0.6307223683557819]
std_gauss_filter = 2.604
noise_var = 4.164**2

# Methods to generate background and artery colors.
def get_artery_ratio():
    ratio_dist = lognorm(*ratio_lognorm_param)
    guess = ratio_dist.rvs(1)
    return guess


# Naive method of generating background and artery colors.
# returns background and artery colors as tuple
def gen_art_color(background):
    # compute ratio
    artery_ratio = get_artery_ratio()
    # compute artery color using ratio and mean background color
    bg_mean_col = np.mean(background)
    art_col = artery_ratio * bg_mean_col
    return art_col

def scale_img(arr):
    return arr/(np.max(arr)/1)


def gen_noise_map(arr, noise_std, noise_mu):
    dimX, dimY = len(arr), len(arr[0])
    #noise_map = np.zeros((dimX, dimY))
    noise_mask = np.random.normal(noise_mu, noise_std, (dimX, dimY))
    for i in range(dimX):
        for j in range(dimY):
            noise_mask[i, j] = arr[i, j]*noise_mask[i, j]
    return noise_mask

def scale_blur_art(arr, std_filter):
    blurred_art = gaussian_filter(np.array(arr, dtype="float"), std_filter)
    scaled_blur_art = scale_img(blurred_art)
    return scaled_blur_art

def put_together(bg, mult_map, noise_map, art_col):
    dimX, dimY = len(bg), len(bg[0])
    ret_img = np.zeros((dimX, dimY))
    for i in range(dimX):
        for j in range(dimY):
            ret_img[i, j] = bg[i, j] * (1-mult_map[i, j]) + (art_col*mult_map[i, j] + noise_map[i, j])

    #Ensure it is int array
    ret_img = np.array(ret_img, dtype="int32")
    return ret_img

# works when label is 2D array of 1'es and 0'es.
# outputs input represented as 2D float array between 0 and 1.
# Backgrounds should be a dataset
def labelToInput(label, background):
    arr = np.array(label.copy()).astype(float)
    # artery color. For now, done in naive manner.
    art_col = gen_art_color(background)
    # apply blur scaled to 0-1
    scaled_blur_art = scale_blur_art(arr, std_gauss_filter)

    # gen noise_map
    noise_map = gen_noise_map(scaled_blur_art, np.sqrt(noise_var), 0)

    # do something
    background = background

    img = put_together(background, scaled_blur_art, noise_map, art_col)

    #add bias field last!
    img = add_bias_field(img)

    return img

"""

def addBlur(img, std):
    blurred_img = gaussian_filter(img, std)
    return blurred_img

#Adds gaussian noise to a 2D numpy arr
def addNoise(arr):
    dimX = len(arr[0])
    #generate arr w noise
    noise_array = poisson.rvs(lambdaPois,size=dimX*dimX)
    noise_mask = np.reshape(noise_array,(dimX,dimX))
    noisy_arr = arr + noise_mask
    #noisy_arr = arr + noise
    #ensures output stays within rgb borders (as input)
    noisy_arr[noisy_arr > 255] = 255
    noisy_arr[noisy_arr < 0] = 0
    #makes arr discrete. Akin to actual data
    #ret_arr = np.array(noisy_arr).astype(int)
    return noisy_arr

"""

"""
dim = 736
#path to local data file
pathDir = "../../../Orig_data/"

artery_im1 = pathDir + r"ImsegmentedPt_02 V_0/Orig/IMG00000_30.tiff"

aim1 = cv2.imread(artery_im1, -1)


#compute colors
back_col, art_col = gen_colors()
print("background color: {0}, artery color: {1}".format(back_col, art_col))


#make a tree. For now constants are hardcoded, this is to be generalized. 
tree1 = Tree(0, 362, 15, 15, 0, 2)
synLab = genTree(tree1, (736, 736))
synInput = labelToInput(synLab)
drawTree(synInput)
#hack to ensure array is int arr not bool arr
#synTree = np.array(genTree(tree1, (736, 736))).astype(int)

#insert artery and background color
#synTree[synTree > 0] = art_col
#synTree[synTree == 0] = back_col
#print(synTree)

#draw input
#plt.imshow(synTree, cmap="gray", vmin=0, vmax=255)
#plt.show()
#draw label
#drawTree(tree1, ((736, 736)))
"""

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