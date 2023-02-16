from TreeLib import Tree
from TreeLib import drawTree, genTree
from genInputFromLabel import labelToInput
import torch
import os
import skimage.io
from torchvision.transforms import ToTensor
import numpy as np
import random

#sets random state of transformations
#ensures 2 different imgs may have same trans applied
def perform_transform(img, seed, trans_dict):
    #set random state
    random.seed(seed)
    torch.manual_seed(seed)
    trans_img = [trans(img) for trans in trans_dict]
    return trans_img

#Dataset class for synthetic data. Needs folder where synthetic data is.
class SynData(torch.utils.data.Dataset):
    def __init__(self, data_dir, label_dir, transforms_both = None, transforms_train = None):
        #save transformations
        self.transforms_both = transforms_both
        self.transforms_train = transforms_train
        #save placement of data and labels
        self.get_data(data_dir, label_dir)

    def get_data(self, data_dir, label_dir):
        files = os.listdir(data_dir)
        files.sort()
        self.data = files
        files_lab = os.listdir(label_dir)
        files_lab.sort()
        self.labs = files_lab
        #if number of element
        if len(self.labs) != len(self.data):
            raise ValueError("Not given the same number of labels and data")

    def __getitem__(self, index):
        #read data
        path_data = os.path.join(self.data_dir, self.data[index])
        data = ToTensor()(skimage.io.imread(path_data))
        #read label
        path_lab = os.path.join(self.lab_dir, self.lab[index])
        lab = ToTensor()(skimage.io.imread(path_lab))

        #seed for transformations
        seed = np.random.randint(0, 2**32)

        #perform transformations
        if self.transforms_train is not None:
            data = perform_transform(data, seed, self.transforms_train)
        if self.transforms_both is not None:
            data = perform_transform(data, seed, self.transforms_both)
            lab = perform_transform(lab, seed, self.transforms_both)

        return data, lab

    def __len__(self):
        return len(self.data)
    


#Dataset class for action angio data
class AngioData(torch.utils.data.Dataset):
    def __init__(self):
        self.num = 1
    def __getitem__(self, index):
        return None
    def __len__(self):
        return -1
