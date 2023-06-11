import numpy as np
import matplotlib.pyplot as plt
import skimage.io

illustrate = False

#
#Image 4_33
path_4_33 = r"Z:\dikuAngiograms\Data\02\ImsegmentedPt_02 V_4\ImsegmentedPt_02 V_4\Orig\IMG00004_33.tiff"
img_4_33 = np.array(skimage.io.imread(path_4_33))

#Define slices
slice1_4_33 = img_4_33[335:336,85:110]
slice2_4_33 = img_4_33[272:273, 438: 467]

path_6_33 = r"Z:\dikuAngiograms\Data\02\ImsegmentedPt_02 V_6\ImsegmentedPt_02 V_6\Orig\IMG00006_33.tiff"
img_6_33 = np.array(skimage.io.imread(path_6_33))

slice_6_33 = img_6_33[605:606, 648:667]

#slices may only be 1d array going from small value -> big value


slices = [slice1_4_33[0], slice2_4_33[0], slice_6_33[0]]

if illustrate:
    for s in slices:
        plt.imshow(np.array([s]), cmap="gray")
        plt.show()