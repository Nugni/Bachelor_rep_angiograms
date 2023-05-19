#%%
import numpy as np
import matplotlib.pyplot as plt
import sys
import os
import skimage.io
import cv2
import torch.utils.data as td

save_path = r"C:\Users\nugni\OneDrive\Skrivebord\Bachelor\git\Bachelor_rep_angiograms\src\images_report"

#go out to src folder
sys.path.append(r'../..')
sys.path.insert(1, r"C:\Users\nugni\OneDrive\Skrivebord\Bachelor\git\Bachelor_rep_angiograms\src\syn_data_functionality")

from data_sets import SynData, BackgroundData
from syn_data_functionality.bias_field import get_bias_field

#bgDataSet = BackgroundData(r"Z:\dikuAngiograms\Projects\Jeppe Filippa Spring 2023\02\Backgrounds")

erda_path = r"Z:\dikuAngiograms\Projects\Jeppe Filippa Spring 2023\02\Data"

dataSet1 = SynData(r"Z:\dikuAngiograms\Projects\Jeppe Filippa Spring 2023\02\Data\ImsegmentedPt_02 V_1\Orig", r"Z:\dikuAngiograms\Projects\Jeppe Filippa Spring 2023\02\Data\ImsegmentedPt_02 V_1\Annotations")
#dataSet3 = SynData(erda_path + r"\ImsegmentedPt_02 V_3\Orig" , erda_path + r"\ImsegmentedPt_02 V_3\Annotations")
dataSet4 = SynData(erda_path + r"\ImsegmentedPt_02 V_4\Orig" , erda_path + r"\ImsegmentedPt_02 V_4\Annotations")
dataSet6 = SynData(erda_path + r"\ImsegmentedPt_02 V_6\Orig" , erda_path + r"\ImsegmentedPt_02 V_6\Annotations")
dataSet7 = SynData(erda_path + r"\ImsegmentedPt_02 V_7\Orig" , erda_path + r"\ImsegmentedPt_02 V_7\Annotations")
dataSet8 = SynData(erda_path + r"\ImsegmentedPt_02 V_8\Orig" , erda_path + r"\ImsegmentedPt_02 V_8\Annotations")

data_sets_lst = [dataSet1, dataSet4, dataSet6, dataSet7, dataSet8]

min_set_len = np.min([len(ds) for ds in data_sets_lst])

#Make balanced data sets
data_subsets = [td.Subset(ds, (np.random.choice(len(ds), min_set_len, replace = False))) for ds in data_sets_lst]

dataSet = td.ConcatDataset(data_subsets)

##%%
#import torchvision.utils
#dloader = td.DataLoader(dataSet8)
#
#it = iter(dloader)
#for i in range(len(it)):
#    imgs, labs = next(it)
#    grid = torchvision.utils.make_grid(imgs) #.numpy()[0] hack to show tensor in plt
#    plt.imshow(grid.numpy()[0], cmap="gray", vmin=0, vmax=1)
#    plt.show()
#    lab_grid = torchvision.utils.make_grid(labs)
#    plt.imshow(lab_grid.numpy()[0], cmap="gray", vmin=0, vmax=1)
#    plt.show()
#

#%%
#a, b = dataSet[len(dataSet)-1]

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

#%%
test_dat, test_lab = dataSet[len(dataSet)-1]
#
art_sample_test, eroted_lab_test = sample_artery(test_dat.numpy()[0], test_lab.numpy()[0], 4, True)
bg_sample_test = sample_background(test_dat.numpy()[0], test_lab.numpy()[0])

bg_col_mean = [[] for elm in range(len(data_subsets))]
art_col_mean =  [[] for elm in range(len(data_subsets))]

#%%
for i in range(len(data_subsets)):
    current_ds = data_subsets[i]
    for j in range(len(current_ds)):
        dat, lab = current_ds[j]
        if np.max(dat.numpy()[0]) <= 10:
            dat = dat.numpy()[0]*(255/np.max(dat.numpy()[0]))
        else:
            dat = dat.numpy()[0]
        bias_field = get_bias_field(dat, lab.numpy()[0])
        dat = dat - bias_field + bias_field.mean()
        art_sample = sample_artery(dat, lab.numpy()[0])
        bg_sample = sample_background(dat, lab.numpy()[0])
        #if np.max(art_sample) <= 20:
        #    print("bg: {0}, art: {1}, dataidx: {2}".format(np.max(bg_sample), np.max(art_sample), i))
        bg_col_mean[i].append(np.mean(bg_sample))
        art_col_mean[i].append(np.mean(art_sample))
        if np.mean(bg_col_mean[i]) <= 10:
            print("ds: {0}, img: {1}".format(i, j))
    print("Finished set {0}".format(i))
#%%

art_bg_ratio = [list(zip(art_col_mean[i], bg_col_mean[i])) for i in range(len(data_subsets))]
#%%
full_arr = [elm for sublst in art_bg_ratio for elm in sublst]

#%%
plt.hist([r[0]/r[1] for r in art_bg_ratio[0]], alpha=0.5, label="set 1")
#plt.hist([r[0]/r[1] for r in art_bg_ratio[1]], alpha=0.5, label="set 3")
plt.hist([r[0]/r[1] for r in art_bg_ratio[1]], alpha=0.5, label="set 4")
plt.hist([r[0]/r[1] for r in art_bg_ratio[2]], alpha=0.5, label="set 6")
plt.hist([r[0]/r[1] for r in art_bg_ratio[3]], alpha=0.5, label="set 7")
plt.hist([r[0]/r[1] for r in art_bg_ratio[4]], alpha=0.5, label="set 8")
plt.legend(loc="upper right")
plt.xlabel("Ratio")
plt.ylabel("Obsevations")
plt.title("Ratio between mean artery and background colors in different sets")
#plt.show()
plt.savefig(save_path + "/" + "color_ratio_in_sets.PNG")

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
ax[0].imshow(test_dat.numpy()[0], cmap="gray")
ax[1].title.set_text("Annotated label")
ax[1].imshow(test_lab.numpy()[0], cmap="gray")
ax[2].title.set_text("Eroted label")
ax[2].imshow(eroted_lab_test, cmap="gray")
plt.xlabel("Visualization of ground truth, annotation, and eroded annotation")
plt.savefig(save_path + "/" + "erode_annot_visualization.PNG")
#plt.show()

fig, ax = plt.subplots(1,2)
ax[0].hist(art_sample_test)
ax[1].hist(bg_sample_test)
ax[0].title.set_text("artery")
ax[1].title.set_text("background")
plt.xlabel("Color values")
plt.savefig(save_path + "/" + "col_bg_art_example.PNG")
plt.show()

#print(np.mean(bg_col_mean))
#print(np.mean(art_col_mean))

fig, ax = plt.subplots(1,3)
ax[0].hist([elm for sublst in art_col_mean for elm in sublst], bins=10)
ax[1].hist([elm for sublst in bg_col_mean for elm in sublst], bins=10)
ax[2].hist([r[0]/r[1] for r in full_arr], bins=10)
ax[0].title.set_text("artery mean")
ax[1].title.set_text("background mean")
ax[2].title.set_text("ratio of bg and art")
plt.xlabel("Mean artery and background colors, and ratio between these")
plt.savefig(save_path + "/" + "mean_bg_art_colors.PNG")
#plt.show()


# %%
print("Assume ratio between bg mean and artery mean is gauss distributed. Then we have mean: {:.3f} and std: {:.3f}".format(np.mean([r[0]/r[1] for r in full_arr]), np.std([r[0]/r[1] for r in full_arr])))
# %%
print("Further min ratio is: {:.3f} and max ratio is: {:.3f}".format(np.min([r[0]/r[1] for r in full_arr]),np.max([r[0]/r[1] for r in full_arr])))

# %%
import matplotlib.pyplot as plt
#for illustration/modelling purposes
def gauss(x, A, x0, var):
    return A * np.exp(-(x - x0)**2 / (2 * var))

def lognorm(x, A, x0, var):
    return gauss(np.log(x), A, x0, var)

from scipy.optimize import curve_fit

plt.figure()
(y, bins, patches) = plt.hist([r[0]/r[1] for r in full_arr], bins=10)
center_bins = bins[:-1] + np.diff(bins) / 2
popt, pcov = curve_fit(lognorm, center_bins, y, p0 = [2, np.mean([r[0]/r[1] for r in full_arr]), np.var([r[0]/r[1] for r in full_arr])])
x_for_ill = np.linspace(center_bins[0], center_bins[-1], 4000)
plt.plot(x_for_ill, gauss(x_for_ill, *popt), label="Fitted lognormal function")
plt.xlabel("Ratio between artery mean and background mean (w/o bias field)")
plt.ylabel("Number of observations")
plt.legend()
plt.savefig(save_path + "/" + "color_ratio_w_lognorm_fit.PNG")
#plt.show()

print("parameters for lognormal fit of ratios are: {0}, {1}, {2}".format(*popt[0], *popt[1], *popt[2]))


# %%
