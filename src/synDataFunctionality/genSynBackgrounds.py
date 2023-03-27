from skimage.restoration import inpaint_biharmonic
import cv2
import numpy as np

# function which given input image, label and degree to thicken line with
# outputs a background with blod vessel removed.
# Image(np.array), label(np.array) and kernel dimensionality
def img_to_background(image, label, kernel_dim):
    kernel = np.ones((kernel_dim, kernel_dim), np.uint8)
    thick_lab = cv2.dilate(label, kernel=kernel, iterations=1)
    background_out = inpaint_biharmonic(image, thick_lab)
    return background_out