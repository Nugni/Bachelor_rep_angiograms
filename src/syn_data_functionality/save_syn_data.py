from syn_data_functionality.tree_lib import Tree, gen_tree
from syn_data_functionality.gen_input_from_label import labelToInput
from data_sets import BackgroundData
import syn_data_functionality.bias_field as bias_field
import numpy as np
import skimage.io
import os
import glob
import torch.utils.data as td
import random as rnd
import math

#startX = 200#5
startY = 0#36 0
#startAngle = 1.2 # 0
starWidth = 20
stopWidth = 2
startLength = 20
bifurcProb = 0.3 #should be changed to prop dist dependent on number of 'straigt' lines

import random as rnd
def make_ran_init_x_and_angle():
    init = rnd.randint(0,2)
    if init == 1:
        startX = rnd.randint(50,100)
        startAngle = rnd.uniform(math.pi/8,math.pi/8*3)
    else:
        startX = rnd.randint(350,400)
        startAngle = rnd.uniform(math.pi/8*3,math.pi/8*5)
    return startX, startAngle

# generates num_samples synthetic trees and labels using tree_params_lst.
# Saves these in data_dir and lab_dir folders respectively.
def gen_syn_data(input_dir, label_dir, backgrounds_data_set, data_dims, num_samples, bias_field_data_set):
    order_66(input_dir=input_dir, label_dir=label_dir)
    # load faktiske baggrunde
    # lav en dataloader, der shuffler(?). Behøver vi faktisk ikke
    len_bg_ds = len(backgrounds_data_set)
    for i in range(num_samples):
        #Get random background from dataset
        idx = rnd.randint(0, len_bg_ds-1)
        bg = backgrounds_data_set[idx].numpy()[0] *255

        # Apply bias field
        bg = bias_field.add_bias_field(bg)

        # label init parameters
        startX, startAngle = make_ran_init_x_and_angle()
        lst = [startX, startY, starWidth, startLength, startAngle, stopWidth, bifurcProb]

        # make label
        tree = Tree(*lst)#tree_params_lst)
        syn_lab = gen_tree(tree, data_dims)
        #make data
        data = labelToInput(syn_lab, bg)
        skimage.io.imsave(input_dir +"/data_{0}.tiff".format(i), data, check_contrast=False)
        skimage.io.imsave(label_dir +"/lab_{0}.tiff".format(i), np.array(syn_lab).astype(int), check_contrast=False)

# deletes all synthetic data files if data==True, and all labels if labels==True
def order_66(input_dir=None, label_dir=None):
    if input_dir is not None:
        files = glob.glob(input_dir + "/*")
        for f in files:
            os.remove(f)
    if label_dir is not None:
        files = glob.glob(label_dir + "/*")
        for f in files:
            os.remove(f)

#read images.
def readSynDat(dir_name):
    return