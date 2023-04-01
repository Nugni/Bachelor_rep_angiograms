#%%
import numpy as np
import matplotlib.pyplot as plt
import sys
import os
import skimage.io
import cv2

#go out to src folder
sys.path.append(r'../..')

from data_sets import SynData, BackgroundData

#bgDataSet = BackgroundData(r"Z:\dikuAngiograms\Projects\Jeppe Filippa Spring 2023\02\Backgrounds")

dataSet = SynData(r"Z:\dikuAngiograms\Projects\Jeppe Filippa Spring 2023\02\Data\Orig", r"Z:\dikuAngiograms\Projects\Jeppe Filippa Spring 2023\02\Data\Annotations")

a, b = dataSet[len(dataSet)-1]

#plt.imshow(a.numpy()[0])
#plt.show()
#plt.imshow(b.numpy()[0])
#plt.show()

def fix_data(lab, img):
    if np.max(lab) > 1:
        lab[lab < np.max(lab)/2] = 0
        lab[lab > 0] = 1
    if np.max(img) <= 1:
        img = img*255
    return np.array(img, dtype="float"), np.array(lab, dtype="float")

# function for shrinking label
def erode_lab(lab, kernel_dim):
    kernel = np.ones((kernel_dim, kernel_dim), np.uint8)
    erosion = cv2.erode(lab, kernel=kernel, iterations=1)
    return erosion

# enlarge label
def dilate_lab(lab, kernel_dim):
    kernel = np.ones((kernel_dim, kernel_dim), np.uint8)
    dilation = cv2.dilate(lab, kernel=kernel, iterations=1)
    return dilation

# Sample artery color pixel values from an image
def sample_artery(img, lab, kernel_dim=4, to_illustrate=False):
    img, lab = fix_data(lab, img)
    big_lab = erode_lab(lab, kernel_dim)
    indices = np.array(big_lab, dtype="bool")
    artery_values = img[indices]
    if to_illustrate:
        return artery_values, big_lab
    return artery_values

# sample background pixel values from an image
def sample_background(img, lab, kernel_dim=10):
    img, lab = fix_data(lab, img)
    big_lab = dilate_lab(lab, kernel_dim)
    indices_neg = np.array(big_lab, dtype="bool")
    indices = ~indices_neg
    bg_values = img[indices]
    return bg_values

test_dat, test_lab = dataSet[len(dataSet)-1]
#
art_sample_test, eroted_lab_test = sample_artery(test_dat.numpy()[0], test_lab.numpy()[0], 4, True)
bg_sample_test = sample_background(test_dat.numpy()[0], test_lab.numpy()[0])

bg_col_mean = []
art_col_mean = []

for i in range(len(dataSet)):
    dat, lab = dataSet[i]
    art_sample = sample_artery(dat.numpy()[0], lab.numpy()[0])
    bg_sample = sample_background(dat.numpy()[0], lab.numpy()[0])
    bg_col_mean.append(np.mean(bg_sample))
    art_col_mean.append(np.mean(art_sample))
#%%

art_bg_ratio = tuple(zip(art_col_mean, bg_col_mean))
#%%
full_arr = np.array(art_bg_ratio)

set4, set6, set7, set8 = full_arr[0:8], full_arr[8:8+12], full_arr[8+12:8+12+7], full_arr[8+12+7:-1]
#%%
plt.hist([r[0]/r[1] for r in set4], alpha=0.5, label="set 4")
plt.hist([r[0]/r[1] for r in set6], alpha=0.5, label="set 6")
plt.hist([r[0]/r[1] for r in set7], alpha=0.5, label="set 7")
plt.hist([r[0]/r[1] for r in set8], alpha=0.5, label="set 8")
plt.legend(loc="upper right")
plt.show()

#%%
"""
for i in range(len(dataSet)):
    dat, lab = dataSet[i]
    dat, lab = dat.numpy()[0], dat.numpy()[0]
    art_sample = sample_artery(dat.copy(), lab.copy())
    bg_sample = sample_background(dat.copy(), lab.copy())
    print(bg_sample)
    bg_col_mean.append(np.mean(bg_sample))
    art_col_mean.append(np.mean(art_sample))
"""

#Visualization of data from example image
fig, ax = plt.subplots(1,3)
ax[0].title.set_text("Ground truth img")
ax[0].imshow(test_dat.numpy()[0])
ax[1].title.set_text("Annotated label")
ax[1].imshow(test_lab.numpy()[0])
ax[2].title.set_text("Eroted label")
ax[2].imshow(eroted_lab_test)
plt.title("Visualization of ground truth, annotation, and eroded annotation")
plt.show()

fig, ax = plt.subplots(1,2)
ax[0].hist(art_sample_test)
ax[1].hist(bg_sample_test)
ax[0].title.set_text("artery colour")
ax[1].title.set_text("background colour")
plt.title("Example histograms of artery color and background color")
plt.show()

print(np.mean(bg_col_mean))
print(np.mean(art_col_mean))

fig, ax = plt.subplots(1,3)
ax[0].hist(art_col_mean, bins=20)
ax[1].hist(bg_col_mean, bins=20)
ax[2].hist([r[0]/r[1] for r in art_bg_ratio], bins=20)
ax[0].title.set_text("mean artery colour")
ax[1].title.set_text("mean background colour")
ax[2].title.set_text("ratio of background and artery colors")
plt.title("Mean artery and background colors")
plt.show()

