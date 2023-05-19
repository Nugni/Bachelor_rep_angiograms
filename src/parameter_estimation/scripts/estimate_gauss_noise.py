import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
sys.path.insert(1, r"C:\Users\nugni\OneDrive\Skrivebord\Bachelor\Bachelor_rep_angiograms\src")
print(os.getcwd())

from parameter_estimation.information.even_color_patches import patches_00, patches_01, patches_02, patches_04, patches_06, patches_07, patches_08



Illustrate = True
save_path = r"C:\Users\nugni\OneDrive\Skrivebord\Bachelor\Bachelor_rep_angiograms\src\images_report"#r"C:\Users\nugni\OneDrive\Skrivebord\Bachelor\git\Bachelor_rep_angiograms\src\images_report"

all_patches = np.concatenate((patches_00, patches_01, patches_02, patches_04, patches_06, patches_07, patches_08))


def gauss(x, A, x0, var):
    return A * np.exp(-(x - x0)**2 / (2 * var))

#We normalize data by subtracting patch mean from each patch
def flatten_and_normalize(patch_lst):
    retLst = [l.flatten() - np.mean(l) for l in patch_lst]
    retLst = [l for sublist in retLst for l in sublist]
    return retLst

tot_data = flatten_and_normalize(all_patches)
patches_indi = [patches_00, patches_04, patches_06]


#IMPORTANT_DATA
gauss_noise_std = np.std(tot_data)
gauss_noise_mu = np.mean(tot_data)
print(np.mean([np.std(p) for p in all_patches]))

#gets center of bins, given bin edges
def get_center_bins(bins_edges):
    return bins_edges[:-1] + np.diff(bins_edges) / 2

if Illustrate:
    plt.title("Pixel values from all patches, normalized")
    (y, bins, patches) = plt.hist(tot_data, bins=20)
    center_bins = get_center_bins(bins)
    popt, pcov = curve_fit(gauss, center_bins, y, p0 = [np.max(y), 0, 20])
    x_for_ill = np.linspace(center_bins[0], center_bins[-1], 4000)
    plt.plot(x_for_ill, gauss(x_for_ill, *popt))
    plt.xlabel("Normalized pixel value")
    plt.ylabel("Number of observations")
    plt.savefig(save_path + "/" + "hist_noise.PNG")

    fig, ax = plt.subplots(1,len(patches_indi))
    for i in range(len(patches_indi)):
        ax[i].hist(flatten_and_normalize(patches_indi[i]))
    plt.show()