import sys

sys.path.insert(1, r"C:\Users\jeppe\Desktop\Bachelor_rep_angiograms\src")

from genSynBackgrounds import make_backgrounds
from saveSynData import order_66

order_66(r"C:\Users\jeppe\Desktop\Data\Backgrounds")

make_backgrounds(
    r"C:\Users\jeppe\Desktop\Data\DataForBackgrounds\Orig",
    r"C:\Users\jeppe\Desktop\Data\DataForBackgrounds\Annotations",
    r"C:\Users\jeppe\Desktop\Data\Backgrounds",
    "test_background"
)

"""For pushing to ERDA"""
"""
make_backgrounds(
    r"Z:\dikuAngiograms\Projects\Jeppe Filippa Spring 2023\02\DataForBackgrounds\Orig",
    r"Z:\dikuAngiograms\Projects\Jeppe Filippa Spring 2023\02\DataForBackgrounds\Annotations",
    r"Z:\dikuAngiograms\Projects\Jeppe Filippa Spring 2023\02\Backgrounds",
    "background"
)
"""