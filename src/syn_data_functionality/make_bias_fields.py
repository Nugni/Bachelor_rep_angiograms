import sys

sys.path.insert(1, r"C:\Users\jeppe\Desktop\Bachelor_rep_angiograms\src")
sys.path.insert(1, r"C:\Users\jeppe\Desktop\Bachelor_rep_angiograms\src\syn_data_functionality")

import bias_field
from save_syn_data import order_66

order_66(r"C:\Users\jeppe\Desktop\Data\bias_fields_model")
order_66(r"C:\Users\jeppe\Desktop\Data\bias_fields_report")
order_66(r"C:\Users\jeppe\Desktop\Data\unbiased_images_model")
order_66(r"C:\Users\jeppe\Desktop\Data\unbiased_images_report")

bias_field.save_new_images(
    r"C:\Users\jeppe\Desktop\Data\DataForBackgrounds\Orig",
    r"C:\Users\jeppe\Desktop\Data\DataForBackgrounds\Annotations",
    r"C:\Users\jeppe\Desktop\Data\bias_fields_model",
    bias_field.get_bias_field_paths,
    "bias_field"
)

bias_field.save_new_images(
    r"C:\Users\jeppe\Desktop\Data\DataForBackgrounds\Orig",
    r"C:\Users\jeppe\Desktop\Data\DataForBackgrounds\Annotations",
    r"C:\Users\jeppe\Desktop\Data\bias_fields_report",
    bias_field.get_bias_field_paths,
    "bias_field_report",
    True
)

bias_field.save_new_images(
    r"C:\Users\jeppe\Desktop\Data\DataForBackgrounds\Orig",
    r"C:\Users\jeppe\Desktop\Data\DataForBackgrounds\Annotations",
    r"C:\Users\jeppe\Desktop\Data\unbiased_images_model",
    bias_field.get_unbiased_image_paths,
    "unbiased_image"
)

bias_field.save_new_images(
    r"C:\Users\jeppe\Desktop\Data\DataForBackgrounds\Orig",
    r"C:\Users\jeppe\Desktop\Data\DataForBackgrounds\Annotations",
    r"C:\Users\jeppe\Desktop\Data\unbiased_images_report",
    bias_field.get_unbiased_image_paths,
    "unbiased_image_report",
    True
)