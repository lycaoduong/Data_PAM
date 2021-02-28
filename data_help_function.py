from nptdms import TdmsFile
import numpy as np

from scipy.signal import hilbert


def TDMS_Info(tdms_file_name, hilbert = True):
    tdms_file = TdmsFile.read(tdms_file_name)
    for group in tdms_file.groups():
        group_name = group.name
    channels = np.array(tdms_file.groups())
    print("Group_name: ", group_name)
    print("Channels_name: ", channels)
    num_Ascan = channels.shape[1]
    record_length = tdms_file[group_name][channels[0, 0]][:].shape[0]
    bscan = np.zeros([record_length,num_Ascan])
    print("Data_size: ", bscan.shape)
    for ascan in range(num_Ascan):
        raw_data_channel = tdms_file[group_name][channels[0, ascan]]
        raw_data = raw_data_channel[:]
        if hilbert==True:
            bscan[:, ascan] = hilbert_scan(raw_data)
        else:
            bscan[:, ascan] = raw_data
    return bscan

def hilbert_scan(raw_data):
    analytic_signal = hilbert(raw_data)
    amplitude_envelope = np.abs(analytic_signal)
    return amplitude_envelope

def scale_to_255(data, min, max):
    scale_data = ((data - min) * (1 / (max - min) * 255))
    scale_data[scale_data<0] = 0
    scale_data[scale_data>255] = 255
    return scale_data

def read_bscan(path_bscan, num_bscan, file_name, reverse = True, hilbert = True):
    _data = []
    cscan = []
    tof_img = []
    for b in range(num_bscan):
        print("Bscan:", b+1)
        bscan = TDMS_Info(path_bscan + file_name + "%s.tdms" %(b+1), hilbert)
        if reverse == True:
            if b%2 ==0:
                _data.append(bscan)
                cscan.append(np.amax(bscan, axis = 0))
                tof_img.append(np.argmax(bscan, axis = 0))
            else:
                bscan = np.flip(bscan, axis = 1)
                _data.append(bscan)
                cscan.append(np.amax(bscan, axis = 0))
                tof_img.append(np.argmax(bscan, axis=0))
        else:
            _data.append(bscan)
            cscan.append(np.amax(bscan, axis = 0))
            tof_img.append(np.argmax(bscan, axis=0))
    _data = np.array(_data)
    cscan = np.array(cscan)
    tof_img   = np.array(tof_img)
    return _data, cscan, tof_img






