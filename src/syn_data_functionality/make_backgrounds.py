import sys

#sys.path.insert(1, r"C:\Users\jeppe\Desktop\Bachelor_rep_angiograms\src")
sys.path.insert(1, r"C:\Users\nugni\OneDrive\Skrivebord\Bachelor\git\Bachelor_rep_angiograms\src")


from gen_syn_backgrounds import make_backgrounds
from save_syn_data import order_66



save_path = r"Z:\dikuAngiograms\Projects\Jeppe Filippa Spring 2023\02\Backgrounds"
data_path = r"Z:\dikuAngiograms\Projects\Jeppe Filippa Spring 2023\02\Data"


#Make background based on angle 0
# order_66(save_path + r"\02 V_0")
# make_backgrounds(
#     data_path + r"\ImsegmentedPt_02 V_0\Orig",
#     data_path + r"\ImsegmentedPt_02 V_0\Annotations",
#     save_path + r"\02 V_0",
#     "bg_V_0_", 
#     num_bg = 3
#     )

#Make background based on angle 1
order_66(save_path + r"\02 V_1")
make_backgrounds(
    data_path + r"\ImsegmentedPt_02 V_1\Orig",
    data_path + r"\ImsegmentedPt_02 V_1\manual",
    save_path + r"\02 V_1",
    "bg_V_1_",
    num_bg = 3
    )

#Make background based on angle 3
order_66(save_path + r"\02 V_3")
make_backgrounds(
    data_path + r"\ImsegmentedPt_02 V_3\Orig",
    data_path + r"\ImsegmentedPt_02 V_3\manual",
    save_path + r"\02 V_3",
    "bg_V_3_",
    num_bg=6
    )

#Make background based on angle 4
#order_66(save_path + r"\02 V_4")
#make_backgrounds(
#    data_path + r"\ImsegmentedPt_02 V_4\Orig",
#    data_path + r"\ImsegmentedPt_02 V_4\Annotations",
#    save_path + r"\02 V_4",
#    "bg_V_4_",
#    num_bg =4
#    )

#Make background based on angle 6
order_66(save_path + r"\02 V_6")
make_backgrounds(
    data_path + r"\ImsegmentedPt_02 V_6\Orig",
    data_path + r"\ImsegmentedPt_02 V_6\Annotations",
    save_path + r"\02 V_6",
    "bg_V_6_",
    num_bg =7
    )

#Make background based on angle 7
order_66(save_path + r"\02 V_7")
make_backgrounds(
    data_path + r"\ImsegmentedPt_02 V_7\Orig",
    data_path + r"\ImsegmentedPt_02 V_7\Annotations",
    save_path + r"\02 V_7",
    "bg_V_7_",
    num_bg =7
    )

#Make background based on angle 8
order_66(save_path + r"\02 V_8")
make_backgrounds(
    data_path + r"\ImsegmentedPt_02 V_8\Orig",
    data_path + r"\ImsegmentedPt_02 V_8\Annotations",
    save_path + r"\02 V_8",
    "bg_V_8_",
    num_bg =3
    )

#make_backgrounds(
#    r"Z:\dikuAngiograms\Projects\Jeppe Filippa Spring 2023\02\DataForBackgrounds\Orig",
#    r"Z:\dikuAngiograms\Projects\Jeppe Filippa Spring 2023\02\DataForBackgrounds\Annotations",
#    r"Z:\dikuAngiograms\Projects\Jeppe Filippa Spring 2023\02\Backgrounds",
#    "background"
#)