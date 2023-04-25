import sys
sys.path.insert(1, r"C:\Users\jeppe\Desktop\Bachelor_rep_angiograms\src")
sys.path.insert(1, r"C:\Users\jeppe\Desktop\Bachelor_rep_angiograms\src\syn_data_functionality")

import bias_field
from save_syn_data import order_66

# TODO: Update paths, so they can take from a given path

order_66(r"Z:\dikuAngiograms\Projects\Jeppe Filippa Spring 2023\02\bias_fields_model")
order_66(r"Z:\dikuAngiograms\Projects\Jeppe Filippa Spring 2023\02\bias_fields_report")
order_66(r"Z:\dikuAngiograms\Projects\Jeppe Filippa Spring 2023\02\unbiased_images_model")
order_66(r"Z:\dikuAngiograms\Projects\Jeppe Filippa Spring 2023\02\unbiased_images_report")

bias_field.save_new_images(
    r"Z:\dikuAngiograms\Projects\Jeppe Filippa Spring 2023\02\DataForBackgrounds\Orig",
    r"Z:\dikuAngiograms\Projects\Jeppe Filippa Spring 2023\02\DataForBackgrounds\Annotations",
    r"Z:\dikuAngiograms\Projects\Jeppe Filippa Spring 2023\02\bias_fields_model",
    bias_field.get_bias_field_paths,
    "bias_field"
)

bias_field.save_new_images(
    r"Z:\dikuAngiograms\Projects\Jeppe Filippa Spring 2023\02\DataForBackgrounds\Orig",
    r"Z:\dikuAngiograms\Projects\Jeppe Filippa Spring 2023\02\DataForBackgrounds\Annotations",
    r"Z:\dikuAngiograms\Projects\Jeppe Filippa Spring 2023\02\bias_fields_report",
    bias_field.get_bias_field_paths,
    "bias_field_report",
    True
)

bias_field.save_new_images(
    r"Z:\dikuAngiograms\Projects\Jeppe Filippa Spring 2023\02\DataForBackgrounds\Orig",
    r"Z:\dikuAngiograms\Projects\Jeppe Filippa Spring 2023\02\DataForBackgrounds\Annotations",
    r"Z:\dikuAngiograms\Projects\Jeppe Filippa Spring 2023\02\unbiased_images_model",
    bias_field.get_unbiased_image_paths,
    "unbiased_image"
)

bias_field.save_new_images(
    r"Z:\dikuAngiograms\Projects\Jeppe Filippa Spring 2023\02\DataForBackgrounds\Orig",
    r"Z:\dikuAngiograms\Projects\Jeppe Filippa Spring 2023\02\DataForBackgrounds\Annotations",
    r"Z:\dikuAngiograms\Projects\Jeppe Filippa Spring 2023\02\unbiased_images_report",
    bias_field.get_unbiased_image_paths,
    "unbiased_image_report",
    True
)