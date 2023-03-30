from SynDataFunctionality.TreeLib import Tree, gen_tree
from SynDataFunctionality.genInputFromLabel import labelToInput
from DataSets import BackgroundData
import numpy as np
import skimage.io
import os
import glob
import torch.utils.data as td
import random as rnd

# generates num_samples synthetic trees and labels using tree_params_lst.
# Saves these in data_dir and lab_dir folders respectively.
def gen_syn_data(input_dir, label_dir, backgrounds_data_set, tree_params_lst, data_dims, num_samples):
    order_66(input_dir=input_dir, label_dir=label_dir)
    # load faktiske baggrunde
    # lav en dataloader, der shuffler(?). Behøver vi faktisk ikke
    len_bg_ds = len(backgrounds_data_set)
    for i in range(num_samples):
        #Get random background from dataset
        idx = rnd.randint(0, len_bg_ds-1)
        bg = backgrounds_data_set[idx].numpy()[0] *255
        #make label
        tree = Tree(*tree_params_lst)
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