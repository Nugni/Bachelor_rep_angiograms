import sys

sys.path.insert(1, r"C:\Users\jeppe\Desktop\Bachelor_rep_angiograms\src")
sys.path.insert(1, r"C:\Users\jeppe\Desktop\Bachelor_rep_angiograms\src\syn_data_functionality")

from bias_field import save_bias_fields
from save_syn_data import order_66

order_66(r"C:\Users\jeppe\Desktop\Data\bias_fields")

save_bias_fields(
    r"C:\Users\jeppe\Desktop\Data\DataForBackgrounds\Orig",
    r"C:\Users\jeppe\Desktop\Data\DataForBackgrounds\Annotations",
    r"C:\Users\jeppe\Desktop\Data\bias_fields",
    "test_background"
)