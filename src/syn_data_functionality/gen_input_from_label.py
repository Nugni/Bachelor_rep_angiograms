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
