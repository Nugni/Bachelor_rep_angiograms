import numpy as np
import skimage.io
import matplotlib.pyplot as plt


path_4_33 = r"Z:\dikuAngiograms\Data\02\ImsegmentedPt_02 V_4\ImsegmentedPt_02 V_4\Orig\IMG00004_33.tiff"
path_6_02 = r"Z:\dikuAngiograms\Data\02\ImsegmentedPt_02 V_6\ImsegmentedPt_02 V_6\Orig\IMG00006_02.tiff"
path_0_07 = r"Z:\dikuAngiograms\Data\02\ImsegmentedPt_02 V_0\Orig\IMG00000_07.tiff"

img_0_07 = np.array(skimage.io.imread(path_0_07))
img_6_02 = np.array(skimage.io.imread(path_6_02))
img_4_33 = np.array(skimage.io.imread(path_4_33))


p1_0_07 = img_0_07[25:35, 250+125:450-15]
p2_0_07 = img_0_07[610+20:610+40, 20+2:20+25]
p3_0_07 = img_0_07[100+40:200, 250+59:250+66]

p1_4_33 = img_4_33[85:100, 100: 130]
p2_4_33 = img_4_33[55:62, 110: 130]
p3_4_33 = img_4_33[65:72, 140: 150]
p4_4_33 = img_4_33[330:343, 105: 115]
p5_4_33 = img_4_33[330:350, 55: 72]

p1_6_22 = img_6_02[320:340, 380:400]
p2_6_22 = img_6_02[70:100, 285:325]
p3_6_22 = img_6_02[106:130, 220:240]





patches_00 = [p1_0_07, p2_0_07, p3_0_07]
patches_01 = []
patches_02 = []
patches_04 = [p1_4_33, p2_4_33, p3_4_33, p4_4_33, p5_4_33]
patches_06 = [p1_6_22, p2_6_22, p3_6_22]
patches_07 = []
patches_08 = []