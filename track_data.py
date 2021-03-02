import cv2
from data_help_function import *
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import find_peaks

data_path = "./DATA_Vessel_2020/2019_Soon Woo/Soon Woo/M2/"
data_name = "earmouse2"
num_Bscan = 350
reverse = True
hilbert = False
sub_length = 100
offset = 40

def left_click(event, x, y, flags, param):
    global xp, yp
    if event == cv2.EVENT_LBUTTONDOWN:
        xp, yp = x, y
        print(xp, yp)
        print(img[yp, xp])
        # yp = int(yp/10)
        bscan = data[yp, :, :]
        bscan = scale_to_255(bscan, 0.005, 0.2)
        bscan = cv2.applyColorMap(bscan.astype(np.uint8), cv2.COLORMAP_HOT)
        cv2.imshow('B-Scan', bscan)
        ascan = ascan_plot(data, xp, yp)
        raw_data = ascan_plot(data_o, xp, yp) + 0.08
        peaks, _ = find_peaks(ascan, height=0.015, width=2)

        max_position = peaks[np.argmax(ascan[peaks])]

        ascan_sub = ascan_plot(sub_data, xp, yp)
        peaks_sub, _ = find_peaks(ascan_sub, height=0.015, width=2)

        fig = plt.figure(figsize=(20, 10))
        ax = fig.add_subplot(111)
        ax.plot(ascan)
        ax.plot(raw_data)
        ax.plot(max_position, ascan[max_position], "x")
        ax.text(max_position+offset, ascan[max_position], "Skin")

        if len(peaks_sub) > 0:
            max_position_sub = peaks_sub[np.argmax(ascan_sub[peaks_sub])]
            # sub_position = max_position+peaks_sub[0]+50
            sub_position = max_position + max_position_sub + offset
            print("Sub: ", sub_position)
            ax.plot(sub_position, ascan[sub_position], "o")
            ax.text(sub_position + offset, ascan[sub_position], "Blood Vessel")


        for i in range(len(peaks)):
            ax.text(peaks[i], ascan[peaks][i], str(peaks[i]))
        plt.savefig('ascan.png')
        # ax.set_aspect(1.0/ax.get_data_ratio()*0.2)
        # plt.show()
        ascanimg = cv2.imread("ascan.png")
        cv2.imshow("A-Scan", ascanimg)




# data, cscan, tof = read_bscan(data_path, num_Bscan, data_name, reverse, hilbert)
data_o, cscan_o, tof_o = read_data_from_npy("tu_hand_15khz.npy")
data, cscan, tof = read_data_from_npy("tu_hand_15khz_hilbert.npy")

cscan = scale_to_255(cscan, 0.02, 0.07)
# cscan = cv2.resize(cscan, (cscan.shape[1], num_Bscan*10))
cscan_color = cv2.applyColorMap(cscan.astype(np.uint8), cv2.COLORMAP_HOT)
cv2.imwrite("tu_hand_hilbert.png", cscan_color)

# print(tof.shape)
# tof = cv2.resize(tof.astype(np.uint8), (tof.shape[1], num_Bscan*10), interpolation = cv2.INTER_LINEAR)
# tof = scale_to_255(tof, 80, 130)
# tof = cv2.applyColorMap(tof.astype(np.uint8), cv2.COLORMAP_RAINBOW)
# cv2.imwrite("earmouse2_tof.png", tof)

sub_data, sub_cscan, sub_tof = filter_layer(data, sub_length, offset)
sub_cscan = scale_to_255(sub_cscan, 0.01, 0.05)
sub_cscan_color = cv2.applyColorMap(sub_cscan.astype(np.uint8), cv2.COLORMAP_HOT)
blur = cv2.blur(sub_cscan_color,(1,1))
cv2.imwrite("sublayer.png", blur)
#
sub_tof = scale_to_255(sub_tof, 0, sub_length)
sub_tof_c = cv2.applyColorMap(sub_tof.astype(np.uint8), cv2.COLORMAP_HOT)
cv2.imwrite("subtof.png", sub_tof)
cv2.imwrite("subtof_c.png", sub_tof_c)

global img
img = cscan_color
cv2.namedWindow('C-Scan', cv2.WINDOW_NORMAL)
cv2.namedWindow('C-Scan_2nd', cv2.WINDOW_NORMAL)
cv2.namedWindow('B-Scan', cv2.WINDOW_NORMAL)
cv2.namedWindow('A-Scan', cv2.WINDOW_NORMAL)
cv2.setMouseCallback('C-Scan_2nd', left_click)

while(1):
    cv2.imshow('C-Scan', img)
    cv2.imshow('C-Scan_2nd', blur)
    if cv2.waitKey(20) & 0xFF == 27:
        qu = True
        break
