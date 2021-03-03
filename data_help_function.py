from nptdms import TdmsFile
import numpy as np
from scipy.signal import hilbert
from scipy.signal import find_peaks
import cv2


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

def ascan_plot(data, xp, yp):
    ascan = data[yp, :, xp]
    return ascan

def read_bscan(path_bscan, num_bscan, file_name, reverse = True, hilbert = True):
    _data = []
    cscan = []
    tof_img = []
    for b in range(num_bscan):
        print("Bscan:", b+1)
        bscan = TDMS_Info(path_bscan + file_name + "%s.tdms" %(b+1), hilbert) #Check the first name of Data file
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

def read_data_from_npy(file):
    data = np.load(file)
    # data = data[:, 500:, :]
    cscan = []
    tof_img = []
    num_bscan = data.shape[0]
    for b in range(num_bscan):
        cscan.append(np.amax(data[b, :, :], axis=0))
        tof_img.append(np.argmax(data[b, :, :], axis=0))
    cscan = np.array(cscan)
    tof_img = np.array(tof_img)
    return data, cscan, tof_img

def filter_layer(data, sub_data_length, offset):
    num_bscan = data.shape[0]
    num_ascan = data.shape[2]
    record_length = data.shape[1]
    sub_data = np.zeros([num_bscan, sub_data_length, num_ascan])
    cscan = np.zeros([num_bscan, num_ascan])
    tof = np.zeros([num_bscan, num_ascan])
    for b in range(num_bscan):
        print("Bscan: ", b)
        # data[b, :, :] = cv2.blur(data[b, :, :], (3,3))
        for a in range(num_ascan):
            ascan = data[b, :, a]
            peaks, _ = find_peaks(ascan, height=0.015, width=2) #Find Peaks
            if len(peaks)>0:
                max_position = peaks[np.argmax(ascan[peaks])]
                if(max_position<(record_length-sub_data_length-offset)):
                    # print(max_position)
                    sub_data[b, :, a] = ascan[max_position+offset:max_position+sub_data_length+offset]
                    ascan_sub = sub_data[b, :, a]
                    peaks, _ = find_peaks(sub_data[b, :, a], height=0.015, width=2)  # Find Peaks
                    if len(peaks)>0:
                        max_position_sub = peaks[np.argmax(ascan_sub[peaks])]
                        # cscan[b, a] = sub_data[b, :, a][peaks][0] #Find First peak
                        # tof[b, a] = peaks[0]
                        cscan[b, a] = sub_data[b, :, a][max_position_sub]
                        tof[b, a] = max_position_sub
                    else:
                        cscan[b, a] = 0
                        tof[b, a] = 255
                else:
                    cscan[b, a] = 0
                    tof[b, a] = 255
            else:
                cscan[b, a] = 0
                tof[b, a] = 255
    return  sub_data, cscan, tof












