import sys
sys.path.insert(1, r"C:\Users\nugni\OneDrive\Skrivebord\Bachelor\Bachelor_rep_angiograms\src")
import torch.utils.data as td
import numpy as np
from syn_data_functionality.save_syn_data import gen_syn_data
from data_sets import SynData, BackgroundData
import torch.utils.data as td

from torchvision.transforms import RandomRotation, RandomResizedCrop, RandomHorizontalFlip, Resize


# number of samples to generate. Uses 0.6, 0.2, 0.2 split
num_samples = 5000

# directory to save data to. Assumes in this there are sub-directories
# train, test and val
save_dir = r"C:\Users\nugni\OneDrive\Skrivebord\Bachelor\local_data\syn_data\syn_data"



# load bias fields
bfDataSet = BackgroundData(r"Z:\dikuAngiograms\Projects\Jeppe Filippa Spring 2023\02\bias_fields_model")

# transformations for backgrounds
bg_trans = [
    RandomRotation(degrees=(-10, 10)),
    RandomResizedCrop(size=(736, 736), scale=(0.6, 0.95), ratio=(0.9, 1.1)),
    RandomHorizontalFlip(p=0.5),
    Resize(size=(736, 736))
]


# background datasets

#bgDataSet0 = BackgroundData(r"Z:\dikuAngiograms\Projects\Jeppe Filippa Spring 2023\02\Backgrounds\02 V_0", transforms=bg_trans)
bgDataSet1 = BackgroundData(r"Z:\dikuAngiograms\Projects\Jeppe Filippa Spring 2023\02\Backgrounds\02 V_1", transforms=bg_trans)
bgDataSet3 = BackgroundData(r"Z:\dikuAngiograms\Projects\Jeppe Filippa Spring 2023\02\Backgrounds\02 V_3", transforms=bg_trans)
bgDataSet4 = BackgroundData(r"Z:\dikuAngiograms\Projects\Jeppe Filippa Spring 2023\02\Backgrounds\02 V_4", transforms=bg_trans)
bgDataSet6 = BackgroundData(r"Z:\dikuAngiograms\Projects\Jeppe Filippa Spring 2023\02\Backgrounds\02 V_6", transforms=bg_trans)
bgDataSet7 = BackgroundData(r"Z:\dikuAngiograms\Projects\Jeppe Filippa Spring 2023\02\Backgrounds\02 V_7", transforms=bg_trans)
bgDataSet8 = BackgroundData(r"Z:\dikuAngiograms\Projects\Jeppe Filippa Spring 2023\02\Backgrounds\02 V_8", transforms=bg_trans)



#train: 3, 4, 5, val: 7, 8, test: 1
BG = td.ConcatDataset([bgDataSet1, bgDataSet3, bgDataSet4, bgDataSet6, bgDataSet7, bgDataSet8])
#valBG = td.ConcatDataset([bgDataSet7, bgDataSet8])
#testBG = bgDataSet1


gen_syn_data(save_dir +"\syn_input", save_dir + "/syn_label", BG, (736, 736), num_samples, bfDataSet)
print("Synthetic training data generated")
#gen_syn_data(save_dir + "/val/input", save_dir + "/val/label", valBG, (736, 736), val_samples, bfDataSet)
#print("Synthetic validation data generated")
#gen_syn_data(save_dir + "/test/input", save_dir + "/test/label", testBG, (736, 736), test_samples, bfDataSet)
#print("Synthetic testing data generated")
#print("Done! :^)")



