from syn_data_functionality.tree_lib import Tree
from syn_data_functionality.tree_lib import draw_tree, gen_tree
from syn_data_functionality.gen_input_from_label import labelToInput
import torch
import os
import skimage.io
from torchvision.transforms import ToTensor
import numpy as np
import random

#sets random state of transformations
#ensures 2 different imgs may have same trans applied
def perform_transform(img, seed, trans_lst):
    #set random state to ensure same random transformation is performed
    random.seed(seed)
    torch.manual_seed(seed)
    for trans in trans_lst:
        img = trans(img)
    return img

#Dataset class for synthetic data. Needs folder where synthetic data is.
class SynData(torch.utils.data.Dataset):
    def __init__(self, data_dir, label_dir, repeat_channels=False, transforms_both = None, transforms_train = None):
        #save transformations
        self.transforms_both = transforms_both
        self.transforms_train = transforms_train
        self.data_dir = data_dir
        self.lab_dir = label_dir
        self.repeat_channels = repeat_channels
        #save placement of data and labels
        self.get_data()

    def get_data(self):
        files = os.listdir(self.data_dir)
        files.sort()
        self.data = files
        files_lab = os.listdir(self.lab_dir)
        files_lab.sort()
        self.labs = files_lab
        #if number of element
        if len(self.labs) != len(self.data):
            raise ValueError("Not given the same number of labels and data")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        #read data
        path_data = os.path.join(self.data_dir, self.data[idx])
        data = ToTensor()(np.array(skimage.io.imread(path_data)))
        #read label
        path_lab = os.path.join(self.lab_dir, self.labs[idx])
        lab = ToTensor()(skimage.io.imread(path_lab))

        #seed for transformations
        ii32 = np.iinfo(np.int32) #sample from as many ints as possible
        seed = np.random.randint(0, ii32.max)

        #perform transformations
        if self.transforms_train is not None:
            data = perform_transform(data, seed, self.transforms_train)
        if self.transforms_both is not None:
            data = perform_transform(data, seed, self.transforms_both)
            lab = perform_transform(lab, seed, self.transforms_both)

        if self.repeat_channels:
            data = torch.repeat_interleave(data[:, :, :], 3, axis=0)

        return data, lab

#change color of background
def change_bckg_col(img):
    return img


#Data set which given backgrounds (no arteries) adjustments are made to images such that new backgrounds can be made
class BackgroundData(torch.utils.data.Dataset):
    def __init__(self, backg_dir, transforms=None, repeat_channels=False):
        self.transforms = transforms
        self.repeat_channels = repeat_channels
        #save placement of backgrounds
        self.backg_dir = backg_dir
        self.files = os.listdir(backg_dir)

    def __getitem__(self, idx):
        path_data = os.path.join(self.backg_dir, self.files[idx])
        data = ToTensor()(np.array(skimage.io.imread(path_data)))

        data = change_bckg_col(data)

        if self.transforms is not None:
            for trans in self.transforms:
                data = trans(data)

        if self.repeat_channels:
            data = torch.repeat_interleave(data[:, :, :], 3, axis=0)

        return data

    def __len__(self):
        return len(self.files)


#Dataset class for action angio data
class AngioData(torch.utils.data.Dataset):
    def __init__(self):
        self.num = 1
    def __getitem__(self, index):
        return None
    def __len__(self):
        return -1
