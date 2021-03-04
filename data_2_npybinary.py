from data_help_function import *
import numpy as np

data_path = "./DATA_Vessel_2020/2019_Soon Woo/Soon Woo/M2/"
data_name = "earmouse2"
num_Bscan = 350
reverse = True
hilbert = True
min = 0.005
max = 0.2

npy_name = "ha.npy"


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
