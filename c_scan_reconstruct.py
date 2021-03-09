import cv2
from data_help_function import *
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from scipy.signal import find_peaks, peak_widths
from sklearn.decomposition import FastICA
from scipy.fftpack import fft,fftshift

data_path = "./DATA_Vessel_2020/2019_Soon Woo/Soon Woo/M2/"
data_name = "earmouse2"
num_Bscan = 350
reverse = True
hilbert = False
ascan_path = "./data_ascan/hand_tu/2d_ascan_hilbert/"


# data, cscan, tof = read_data_from_npy("ha.npy")
# print(data.shape)
# bscan = 1


# data, cscan, tof = read_bscan(data_path, num_Bscan, data_name, reverse, hilbert)


def ascan_reconstruct(data, path_save, min, max):
    # bscan = data.shape[0]
    # line = np.zeros([bscan, 256, 256])
    bscan = 1
    for y in range(bscan):
        print(y)
        y = y + 50
        for x in range(data.shape[2]):
            ascan = ascan_plot(data, x, y)
            img = ascan_2_img(ascan, min, max)
            file_name = "%sy_%sx_ascan.png" %(y,x)
            cv2.imwrite(path_save + file_name, img)
    print("Finish")

def ascan_plot_test(ascan, find_peak = False, height = 0.015, width = 2):
    fig = plt.figure(figsize=(20, 10))
    ax = fig.add_subplot(111)
    peaks, _ = find_peaks(ascan, height=0.04, width=2)
    print(peaks)
    results_full = peak_widths(ascan, peaks, rel_height=0.5)
    print(results_full[0])
    ax.plot(ascan)
    plt.plot(peaks, ascan[peaks], "x")
    plt.hlines(*results_full[1:], color="C2")
    plt.savefig("abc.png")
    plt.show()

def cscan_reconstruct(cscan):
    cscan = scale_to_255(cscan, 0.02, 0.07)
    cscan = cv2.resize(cscan, (cscan.shape[1], num_Bscan*10))  # Interpolate data
    cscan_color = cv2.applyColorMap(cscan.astype(np.uint8), cv2.COLORMAP_HOT)
    cv2.imwrite("c_scan_reconstruct.png", cscan_color)
    cv2.imshow("CScan", cscan_color)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def img_to_ascan_reconstruct(img, presict = False):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ascan = img_2_ascan(img, predict=presict)
    print(ascan)
    return ascan


def fastICA(img_ascan, n_compo):
    img = cv2.cvtColor(img_ascan, cv2.COLOR_RGB2GRAY)
    ascan = img.T/255
    ica = FastICA(n_components=n_compo)
    S_ = ica.fit_transform(ascan)
    models = [ascan, S_]
    S = S_.T
    return S
# data, cscan, tof = read_data_from_npy("tu_hand_15khz_hilbert.npy")
# ascan_reconstruct(data, ascan_path, 0, 0.08)
img = cv2.imread("./data_ascan/hand_tu/2d_ascan_hilbert/50y_120x_ascan.png")
ascan = img_to_ascan_reconstruct(img)
ascan = ascan/255
hilbert = hilbert_scan(ascan)
plt.plot(ascan)
plt.plot(hilbert+0.5)
plt.show()