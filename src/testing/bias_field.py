from skimage import io, morphology, filters
import numpy as np

impath = r"C:\Users\jeppe\Desktop\Data\6\Orig\IMG00006_28.tiff"
annotpath = r"C:\Users\jeppe\Desktop\Data\6\Annot\I6_028.tiff"

def get_bias_field(real_image_path,real_annotation_path):
    real_image = io.imread(real_image_path)
    real_annotation = io.imread(real_annotation_path)
    real_annotation_mask = real_annotation < 1

    (rows,cols) = real_image.shape
    r, c = np.meshgrid(list(range(rows)), list(range(cols)))
    rMsk = r[real_annotation_mask].flatten()
    cMsk = c[real_annotation_mask].flatten()
    VanderMondeMsk = np.array([rMsk*0+1, rMsk, cMsk, rMsk**2, rMsk*cMsk, cMsk**2]).T
    ValsMsk = real_image[real_annotation_mask].flatten()
    coeff, residuals, rank, singularValues = np.linalg.lstsq(VanderMondeMsk, ValsMsk, rcond=None)
    VanderMonde = np.array([r*0+1, r, c, r**2, r*c, c**2]).T
    J = np.dot(VanderMonde, coeff) # @ operator is a python 3.5 feature!
    J = J.reshape((rows,cols)).T
    return(J)

# Can be deleted, only here for testing
from matplotlib import pyplot as plt, cm
plt.rcParams['image.cmap'] = 'gray' # set default colormap for imshow to be gray
plt.imshow(get_bias_field(impath,annotpath))
plt.show()
