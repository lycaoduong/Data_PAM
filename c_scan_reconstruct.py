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

data, cscan, tof = read_bscan(data_path, num_Bscan, data_name, reverse, hilbert)

cscan = scale_to_255(cscan, 0.02, 0.07)
cscan = cv2.resize(cscan, (cscan.shape[1], num_Bscan*10))  # Interpolate data
cscan_color = cv2.applyColorMap(cscan.astype(np.uint8), cv2.COLORMAP_HOT)
cv2.imwrite("c_scan_reconstruct.png", cscan_color)
cv2.imshow("CScan", cscan_color)
cv2.waitKey(0)
cv2.destroyAllWindows()