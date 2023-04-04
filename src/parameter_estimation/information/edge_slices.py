import numpy as np
import matplotlib.pyplot as plt
import skimage.io

illustrate = False

slices = []

#
#Image 4_33
path_4_33 = r"Z:\dikuAngiograms\Data\02\ImsegmentedPt_02 V_4\ImsegmentedPt_02 V_4\Orig\IMG00004_33.tiff"
img_4_33 = np.array(skimage.io.imread(path_4_33))

#Define slices
slice1_4_33 = img_4_33[335:336,85:110]
slice2_4_33 = img_4_33[272:273, 438: 467]

#slices may only be 1d array going from small value -> big value
slices.append(slice1_4_33[0])
slices.append(slice2_4_33[0])


if illustrate:
    for s in slices:
        plt.imshow(s, cmap="gray")
        plt.show()