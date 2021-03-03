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
sub_length = 200
offset = 30
mouse_move = False
tracking   = False

def left_click(event, x, y, flags, param):
    global xp, yp, mouse_move, tracking
    if event == cv2.EVENT_MOUSEMOVE:
        xp, yp = x, y
        if tracking==True:
            print(xp, yp)
            print(img[yp, xp])
            # yp = int(yp/10)
            bscan = data[yp, :600, :]
            bscan = cv2.blur(bscan, (5, 5))
            bscan = scale_to_255(bscan, 0.005, 0.2)
            bscan = cv2.applyColorMap(bscan.astype(np.uint8), cv2.COLORMAP_HOT)

            verital_cross_section = np.transpose(data[:, :600, xp])
            verital_cross_section = scale_to_255(verital_cross_section, 0.005, 0.2)
            verital_cross_section = cv2.applyColorMap(verital_cross_section.astype(np.uint8), cv2.COLORMAP_HOT)

            ascan = ascan_plot(data, xp, yp)
            raw_data = ascan_plot(data_o, xp, yp) + 0.08
            peaks, _ = find_peaks(ascan, height=0.015, width=2)

            max_position = peaks[np.argmax(ascan[peaks])]
            cv2.circle(bscan, (xp, max_position), 0, (0,255,0), 2)
            cv2.circle(verital_cross_section, (yp, max_position), 0, (0, 255, 0), 2)

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
                cv2.circle(bscan, (xp, sub_position), 0, (0, 255, 0), 2)
                cv2.circle(verital_cross_section, (yp, sub_position), 0, (0, 255, 0), 2)
                ax.hlines(y = (ascan[max_position] + ascan[sub_position])/2, xmin = max_position, xmax = sub_position, color = "C1")
                depth = (sub_position-max_position)*1540*1000/200000000
                ax.text(max_position , (ascan[max_position] + ascan[sub_position])/2, str(depth) + " mm")

            for i in range(len(peaks)):
                ax.text(peaks[i], ascan[peaks][i], str(peaks[i]))
            plt.savefig('ascan.png')
            # ax.set_aspect(1.0/ax.get_data_ratio()*0.2)
            # plt.show()
            ascanimg = cv2.imread("ascan.png")
            cv2.imshow("A-Scan", ascanimg)
            cv2.imshow('B-Scan', bscan)
            cv2.imshow('Cross-Section', verital_cross_section)
        mouse_move = True
    elif event == cv2.EVENT_LBUTTONDOWN:
        tracking = True
    elif event == cv2.EVENT_LBUTTONUP:
        tracking = False





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
sub_cscan = scale_to_255(sub_cscan, 0.015, 0.04)
sub_cscan_color = cv2.applyColorMap(sub_cscan.astype(np.uint8), cv2.COLORMAP_HOT)
blur = cv2.blur(sub_cscan_color,(3,3))
cv2.imwrite("sublayer.png", blur)
blood_vessel = blur.copy()

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
cv2.namedWindow('Cross-Section', cv2.WINDOW_NORMAL)
cv2.setMouseCallback('C-Scan_2nd', left_click)

while(1):
    if mouse_move == True:
        blood_vessel = blur.copy()
        cv2.line(blood_vessel, (xp,0), (xp, blood_vessel.shape[0]), (0,255,0), 1)
        cv2.line(blood_vessel, (0, yp), (blood_vessel.shape[1], yp), (0, 255, 0), 1)
        mouse_move = False
    cv2.imshow('C-Scan', img)
    cv2.imshow('C-Scan_2nd', blood_vessel)
    if cv2.waitKey(20) & 0xFF == 27:
        qu = True
        break
