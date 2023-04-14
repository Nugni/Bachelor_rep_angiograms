import sys

sys.path.insert(1, r"C:\Users\jeppe\Desktop\Bachelor_rep_angiograms\src")

from gen_syn_backgrounds import make_backgrounds
from save_syn_data import order_66

order_66(r"Z:\dikuAngiograms\Projects\Jeppe Filippa Spring 2023\02\Backgrounds")


make_backgrounds(
    r"Z:\dikuAngiograms\Projects\Jeppe Filippa Spring 2023\02\DataForBackgrounds\Orig",
    r"Z:\dikuAngiograms\Projects\Jeppe Filippa Spring 2023\02\DataForBackgrounds\Annotations",
    r"Z:\dikuAngiograms\Projects\Jeppe Filippa Spring 2023\02\Backgrounds",
    "background"
)