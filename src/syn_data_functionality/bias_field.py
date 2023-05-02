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

#Add bias field to unbiased image
def add_bias_field(unbiased_background, bias_dir=r'Z:\dikuAngiograms\Projects\Jeppe Filippa Spring 2023\02\bias_fields_model'):
    bias_path = bias_dir + "\\" + rnd.choice(os.listdir(bias_dir))
    bias_field = io.imread(bias_path)
    #unbiased_background = unbiased_background - unbiased_background.mean()
    return_image = unbiased_background + bias_field - bias_field.mean()
    return return_image
