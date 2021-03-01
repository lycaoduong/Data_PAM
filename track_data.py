import cv2
from data_help_function import *
import matplotlib.pyplot as plt

data_path = "./DATA_Vessel_2020/2019_Soon Woo/Soon Woo/M2/"
data_name = "earmouse2"
num_Bscan = 5
reverse = True
hilbert = True

def left_click(event, x, y, flags, param):
    global xp, yp
    if event == cv2.EVENT_LBUTTONDOWN:
        xp, yp = x, y
        print(xp, yp)
        print(img[yp, xp])
        yp = int(yp/10)
        bscan = data[yp, :, :]
        bscan = scale_to_255(bscan, 0.005, 0.2)
        bscan = cv2.applyColorMap(bscan.astype(np.uint8), cv2.COLORMAP_HOT)
        cv2.imshow('B-scan', bscan)
        ascan = ascan_plot(data, xp, yp)
        plt.plot(ascan)
        plt.show()
        print(tof[yp, xp])




data, cscan, tof = read_bscan(data_path, num_Bscan, data_name, reverse, hilbert)

cscan = scale_to_255(cscan, 0.005, 0.2)
cscan = cv2.resize(cscan, (cscan.shape[1], num_Bscan*10))
cscan_color = cv2.applyColorMap(cscan.astype(np.uint8), cv2.COLORMAP_HOT)
cv2.imwrite("earmouse2_hot.png", cscan_color)

# print(tof.shape)
# tof = cv2.resize(tof.astype(np.uint8), (tof.shape[1], num_Bscan*10), interpolation = cv2.INTER_LINEAR)
# tof = cv2.applyColorMap(tof, cv2.COLORMAP_WINTER)
# cv2.imwrite("earmouse2_tof.png", tof)

global img
img = cscan_color
cv2.namedWindow('C-Scan')
cv2.namedWindow('B-Scan')
cv2.setMouseCallback('C-Scan', left_click)

while(1):
    cv2.imshow('C-Scan', img)
    if cv2.waitKey(20) & 0xFF == 27:
        qu = True
        break
