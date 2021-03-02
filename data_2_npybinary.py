from data_help_function import *
import numpy as np

data_path = "./20210129/"
data_name = "T2_15KhZ"
num_Bscan = 600
reverse = False
hilbert = True
min = 0.005
max = 0.2

npy_name = "tu_hand_15khz_hilbert.npy"


def tdms_2_npy(data_path, num_Bscan, data_name, reverse, hilbert):
    data, cscan, tof = read_bscan(data_path, num_Bscan, data_name, reverse, hilbert)
    # data = scale_to_255(data, min, max)
    # data = data.astype(np.uint8)
    return data

def second_filter(npy_name):
    data, cscan, tof = read_data_from_npy(npy_name)
    sub_data, sub_cscan, sub_tof = filter_layer(data, 50)
    return sub_data

# data = tdms_2_npy(data_path, num_Bscan, data_name, reverse, hilbert)
# np.save('tu_hand_15khz_hilbert.npy', data)

sub_data = second_filter(npy_name)
np.save('tu_remove_skin.npy', sub_data)
