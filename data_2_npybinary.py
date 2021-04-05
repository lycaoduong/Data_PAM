from data_help_function import *
import numpy as np

data_path = "./aTu/TU_Foot_AR_PAM/"
data_name = "tu_foot8_100um"
num_Bscan = 700
reverse = False
hilbert = True
min = 0.005
max = 0.2

npy_name = "./result/earmouse.npy"


def tdms_2_npy(data_path, num_Bscan, data_name, reverse, hilbert):
    data, cscan, tof = read_bscan(data_path, num_Bscan, data_name, reverse, hilbert)
    # data = scale_to_255(data, min, max)
    # data = data.astype(np.uint8)
    return data

def second_filter(npy_name):
    data, cscan, tof = read_data_from_npy(npy_name)
    sub_data, sub_cscan, sub_tof = filter_layer(data, 50)
    return sub_data

data = tdms_2_npy(data_path, num_Bscan, data_name, reverse, hilbert)
np.save(npy_name, data)

# sub_data = second_filter(npy_name)
# np.save('tu_remove_skin.npy', sub_data)
