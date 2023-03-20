from SynDataFunctionality.TreeLib import Tree, gen_tree
from SynDataFunctionality.genInputFromLabel import labelToInput
import numpy as np
import skimage.io
import os
import glob

# generates num_samples synthetic trees and labels using tree_params_lst.
# Saves these in data_dir and lab_dir folders respectively.
def gen_syn_data(input_dir, label_dir, tree_params_lst, data_dims, num_samples):
    order_66(input_dir=input_dir, label_dir=label_dir)
    for i in range(num_samples):
        tree = Tree(*tree_params_lst)
        syn_lab = gen_tree(tree, data_dims)
        data = labelToInput(syn_lab)
        skimage.io.imsave(input_dir +"/data_{0}.tiff".format(i), data)
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