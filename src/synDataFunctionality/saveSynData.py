from synDataFunctionality.TreeLib import Tree, genTree
from synDataFunctionality.genInputFromLabel import labelToInput
import numpy as np
import skimage.io
import os
import glob

#generates num_samples synthetic trees and labels using tree_params_lst.
#Saves these in respectively synInput and synLabel folders.
def genSynDat(data_dir, lab_dir, tree_params_lst, data_dim, num_samples):
    order_66(data_dir=data_dir, lab_dir=lab_dir)
    for i in range(num_samples):
        tree = Tree(*tree_params_lst)
        syn_lab = genTree(tree, data_dim)
        data = labelToInput(syn_lab)
        skimage.io.imsave(data_dir +"/data_{0}.tiff".format(i), data)
        skimage.io.imsave(lab_dir +"/lab_{0}.tiff".format(i), np.array(syn_lab).astype(int), check_contrast=False)

#deletes all synthetic data files if data=True, and all labels if labels=True
def order_66(data_dir=None, lab_dir=None):
    if data_dir is not None:
        files = glob.glob(data_dir + "/*")
        for f in files:
            os.remove(f)
    if lab_dir is not None:
        files = glob.glob(lab_dir + "/*")
        for f in files:
            os.remove(f)

#read images.
def readSynDat(dir_name):
    return