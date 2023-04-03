import os
import sys
from scipy.optimize import curve_fit
import numpy as np
import matplotlib.pyplot as plt
#os.chdir(r'..')
sys.path.append("..\..")
print(os.getcwd())

#Get parameters to use for rest of script
#Get data
from parameter_estimation.information.edge_slices import slices
#whether to generate new figures
gen_figures = False
# Where we wish to save any figures made
file_path = r"C:\Users\nugni\OneDrive\Skrivebord\Bachelor\git\Bachelor_rep_angiograms\src\images_report"

#taken from
#https://stackoverflow.com/questions/60160803/scipy-optimize-curve-fit-for-logistic-function

# functions used for regression
def logifunc(x, A, x0, k, off):
    return A / (1 + np.exp(-k*(x-x0)))+off

def logiDerived(x, A, k, x0):
    return A*k*(np.exp(-k*(x-x0)))/np.power((1+np.exp(-k*(x-x0))), 2)

def gauss(x, A, x0, var):
    return A * np.exp(-(x - x0)**2 / (2 * var))

# fit image to logistic function
def fit_slice_to_logi(imgSlice, guess=None):
    y = imgSlice
    x = np.arange(len(imgSlice))
    popt, pcov = curve_fit(logifunc, x, y, p0=guess)
    return popt

# fit the derivative of a logistic function to a gaussian function, 
# and return that gaussian's parameters.
def find_gauss_from_logi(popt, slicei, illustrate=False, file_name=None, path = None, num_samples=2000):
    x = np.linspace(0, len(slicei)-1, num=num_samples)
    A, x0, k = popt[0], popt[1], popt[2]
    y_logi = logiDerived(x, A, k, x0)

    popt_g, pcov_g = curve_fit(gauss, x, y_logi)

    if illustrate:
        plt.figure()
        plt.title("Gaussian fitted to the derivation of logistic fit")
        plt.xlabel("Pixel value")
        plt.ylabel("Pixel index in slice")

        plt.plot(x, y_logi, label="g : Derivation of f")

        y_gauss = gauss(x, popt_g[0], popt_g[1], popt_g[2])
        plt.plot(x, y_gauss, label="h : Guassian fit of g")

        plt.legend()
        plt.savefig(path + "/" + file_name)

    return popt_g


def illustrate_slice_fit_and_noise(file_name, path, img_slice, mu_gauss, popt_logi):
    x = np.arange(len(img_slice))
    plt.figure()
    plt.scatter(x, img_slice, label="Observations")
    A, off = popt_logi[0], popt_logi[3]
    step_x, step_y = [0, mu_gauss, len(img_slice)-1], [off, off+A, off+A]
    plt.plot(step_x, step_y, drawstyle="steps-post", linestyle='dashed', c='gray', label="Assumed ground truth")
    logi_y = logifunc(x, popt_logi[0], popt_logi[1], popt_logi[2], popt_logi[3])
    plt.plot(logi_y, label="f : Logistic fit of obsevations", c='r')
    plt.xlabel("Slice pixel index")
    plt.ylabel("Pixel value")
    plt.title("Logistic fit of observed blood vessel edge and assumed edge")
    plt.legend()
    plt.savefig(path+"/"+file_name)

stds = []

# Approximate gauss std from logistic fit of all slices
for s in slices:
    guess = [np.max(s)-np.min(s), len(s)/2, 0.3, np.min(s)]
    popt = fit_slice_to_logi(s, guess)
    popt_g = find_gauss_from_logi(popt, s)
    std = np.sqrt(popt_g[2])
    stds.append(std)

# Generate figures using only first slice if we wish
if gen_figures:
    s0 = slices[0]
    guess0 = [np.max(s0)-np.min(s0), len(s0)/2, 0.3, np.min(s0)]
    popt0 = fit_slice_to_logi(s0, guess0)
    popt_g0 = find_gauss_from_logi(popt0, s0, illustrate=True, file_name="gauss_fit_logi_derived.PNG", path = file_path)
    illustrate_slice_fit_and_noise("edge_visualization.PNG" , file_path ,s0, popt_g0[1], popt0)

gauss_filter_std_mean = np.mean(stds)
