from skimage.restoration import inpaint_biharmonic
import cv2
import numpy as np
import os
import skimage.io

gaus_noise_var = 20.8


def rescale_img(img):
    mmin = np.min(img)
    mmax = np.max(img)
    dif = mmax-mmin
    img = (img - mmin) * 255/dif
    return np.array(img, dtype="uint8")

def add_noise(backg, lab_mask):
    dimX, dimY = len(backg), len(backg[0])
    noise_mask = np.random.normal(0, np.sqrt(gaus_noise_var), (dimX, dimY))
    for i in range(dimX):
        for j in range(dimY):
            if lab_mask[i, j] == 1:
                backg[i, j] += noise_mask[i, j]
    return backg


# function which given input image, label and degree to thicken line with
# outputs a background with blod vessel removed.
# Image(np.array), label(np.array) and kernel dimensionality
def img_to_background(image, label, kernel_dim):
    image = image.astype('uint8')
    label = label.astype('uint8')
    kernel = np.ones((kernel_dim, kernel_dim), np.uint8)
    thick_lab = cv2.dilate(label, kernel=kernel, iterations=1)
    #remove blood vessels
    background_no_vessels = inpaint_biharmonic(image, thick_lab)
    #ensure output is between 0 and 255
    background_no_vessels = rescale_img(background_no_vessels)
    print(np.min(background_no_vessels))
    print(np.max(background_no_vessels))
    #Add noise, where no noise is
    background_out = add_noise(background_no_vessels, thick_lab)
    return background_out

#Given directory with original backgrounds, make backgrounds, save new backgrounds in directory.
def make_backgrounds(img_dir, label_dir, save_dir, files_prefix, kernel_dim=8):
    files = os.listdir(img_dir)
    files.sort()
    file_paths = [img_dir + "\\" + str(file) for file in files]
    labels = os.listdir(label_dir)
    labels.sort()
    label_paths = [label_dir + "\\" + str(label) for label in labels]
    assert len(labels) == len(files), "length of labels and backgrounds does not match"
    for i in range(len(labels)):
        img = skimage.io.imread(file_paths[i])
        lab = skimage.io.imread(label_paths[i])
        backg = img_to_background(img, lab, kernel_dim)
        skimage.io.imsave(save_dir + "/" + files_prefix + str(i) + ".tiff", backg)


