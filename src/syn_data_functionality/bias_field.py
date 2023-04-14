from skimage import io, morphology, filters
import numpy as np
import os
import random as rnd

def get_bias_field(real_image, real_annotation):
    real_annotation_mask = real_annotation < 1

    # Smoothing image
    disk_element = morphology.disk(1)
    real_image_filtered = filters.median(real_image, disk_element)

    (rows,cols) = real_image_filtered.shape
    r, c = np.meshgrid(list(range(rows)), list(range(cols)))
    rMsk = r[real_annotation_mask].flatten()
    cMsk = c[real_annotation_mask].flatten()

    VanderMondeMsk = np.array([rMsk*0+1, rMsk, cMsk, rMsk**2, rMsk*cMsk, cMsk**2]).T
    ValsMsk = real_image_filtered[real_annotation_mask].flatten()

    coeff, residuals, rank, singularValues = np.linalg.lstsq(VanderMondeMsk, ValsMsk, rcond=None)
    VanderMonde = np.array([r*0+1, r, c, r**2, r*c, c**2]).T
    J = np.dot(VanderMonde, coeff) # @ operator is a python 3.5 feature!
    J = J.reshape((rows,cols)).T
    return(J)

def get_bias_field_paths(real_image_path, real_annotation_path):
    real_image = io.imread(real_image_path)
    real_annotation = io.imread(real_annotation_path)
    bias_field = get_bias_field(real_image, real_annotation)
    return bias_field


def get_unbiased_image(real_image, real_annotation):
    bias = get_bias_field(real_image, real_annotation)
    return real_image - bias + bias.mean()

def get_unbiased_image_paths(real_image_path, real_annotation_path):
    real_image = io.imread(real_image_path)
    real_annotation = io.imread(real_annotation_path)
    unbiased_image = get_unbiased_image(real_image, real_annotation)
    return unbiased_image


def save_new_images(img_dir, label_dir, save_dir, func, files_prefix, restrict_values=False):
    files = os.listdir(img_dir)
    files.sort()
    file_paths = [img_dir + "\\" + str(file) for file in files]
    labels = os.listdir(label_dir)
    labels.sort()
    label_paths = [label_dir + "\\" + str(label) for label in labels]
    assert len(labels) == len(files), "length of labels and backgrounds does not match"
    for i in range(len(labels)):
        new_image = func(file_paths[i],label_paths[i])
        if restrict_values == True:
            new_image = np.clip(new_image, 0, 255)
            new_image = new_image.astype('uint8')
        io.imsave(save_dir + "/" + files_prefix + str(i) + ".tiff", new_image)

def add_bias_field(image,bias_dir=r'C:\Users\jeppe\Desktop\Data\bias_fields_model'):
    bias_path = bias_dir + "\\" + rnd.choice(os.listdir(bias_dir))
    print(bias_path)
    bias_field = io.imread(bias_path)
    return_image = image + bias_field
    return return_image



bgimg = io.imread(r'C:\Users\jeppe\Desktop\Data\Backgrounds\test_background1.tiff')

from matplotlib import pyplot as plt, cm

for i in range(5):

    syn_bg = add_bias_field(bgimg)
    plt.rcParams['image.cmap'] = 'gray' # set default colormap for imshow to be gray

    fig, ax = plt.subplots(1, 2, figsize=(15,15))

    ax[0].imshow(bgimg)
    ax[1].imshow(syn_bg)
    plt.show()

"""
# Can be deleted, only here for testing
from matplotlib import pyplot as plt, cm
plt.rcParams['image.cmap'] = 'gray' # set default colormap for imshow to be gray

fig, ax = plt.subplots(1, 3, figsize=(15,15))

image = io.imread(impath)
bias_field = get_bias_field(impath,annotpath)
unbiased_image = get_unbiased_image(impath,annotpath)

def get_vals(input):
    return input.min(), input.mean(), input.max()

print([get_vals(x) for x in [image,bias_field,unbiased_image]])

ax[0].imshow(image)
ax[1].imshow(bias_field)
ax[2].imshow(unbiased_image)
plt.show()

#print(unbiased_image)
"""