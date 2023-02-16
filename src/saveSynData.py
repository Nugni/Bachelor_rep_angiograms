from TreeLib import Tree, genTree
from genInputFromLabel import labelToInput
import skimage.io
import os
import glob

#generates num_samples synthetic trees and labels using tree_params_lst.
#Saves these in respectively synInput and synLabel folders.
def genSynDat(dir_name, tree_params_lst, data_dim, num_samples):
    order_66(data=True, labels=True)
    for i in range(num_samples):
        tree = Tree(*tree_params_lst)
        syn_lab = genTree(tree, data_dim)
        data = labelToInput(syn_lab)
        skimage.io.imsave(dir_name +"/synInput/data_{0}.tiff".format(i), data)
        skimage.io.imsave(dir_name +"/synLabel/lab_{0}.tiff".format(i), data)

#deletes all synthetic data files if data=True, and all labels if labels=True
def order_66(data=False, labels=False):
    if data:
        files = glob.glob("SynDat/synInput/*")
        for f in files:
            os.remove(f)
    if labels:
        files = glob.glob("SynDat/synLabel/*")
        for f in files:
            os.remove(f)

#read images.
def readSynDat(dir_name):
    return