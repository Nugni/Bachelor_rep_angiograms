from TreeLib import Tree, genTree
from genInputFromLabel import labelToInput
import skimage.io

def genSynDat(dir_name, tree_params_lst, data_dim, num_samples):
    for i in range(num_samples):
        tree = Tree(*tree_params_lst)
        syn_lab = genTree(tree, data_dim)
        data = labelToInput(syn_lab)
        skimage.io.imsave(dir_name +"data_{0}.tiff".format(i), data)
        skimage.io.imsave(dir_name +"lab_{0}.tiff".format(i), data)

def readSynDat(dir_name):
    return