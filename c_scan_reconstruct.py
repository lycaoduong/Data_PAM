import cv2
from data_help_function import *
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from scipy.signal import find_peaks, peak_widths
from sklearn.decomposition import FastICA
from scipy.fftpack import fft,fftshift
from scipy.io import savemat
import pandas as pd

data_path = "./DATA_Vessel_2020/2019_Soon Woo/Soon Woo/M2/"
data_name = "earmouse2"
num_Bscan = 350
reverse = True
hilbert = False
ascan_path = "./data_ascan/hand_tu/2d_ascan_hilbert/"

mouse_move = False
tracking   = True

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
        y = y + 200
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

def cscan_from_predictdata(data, offset):
    w = data.shape[2]
    h = data.shape[0]
    skin = np.zeros((h, w))
    blood = np.zeros((h, w))
    depth_blood = np.zeros((h, w))

    for b in range(data.shape[0]):
        print("Bscan: ", b)
        for a in range(data.shape[2]):
            ascan = ascan_plot(data, a, b)
            peaks, _ = find_peaks(ascan, height=offset, width=2, distance=35) #35
            if len(peaks) == 1:
                skin[b, a] = ascan[peaks[0]]
            elif len(peaks) == 2:
                skin[b, a] = ascan[peaks[0]]
                blood[b, a] = ascan[peaks[1]]
                depth_blood[b, a] = peaks[1]
            elif len(peaks) > 2:
                skin[b, a] = ascan[peaks[0]]
                blood[b, a] = ascan[peaks[-1]]
                depth_blood[b, a] = peaks[-1]
    depth_code = depth_blood
    depth_code = scale_to_255(depth_code, 200, 1000)
    depth_code[:, 500:999] = scale_to_255(depth_code[:, 500:999], 20, 255)
    # depth_code[depth_code<=50] = 0
    depth_code[depth_code>0] = 255 - depth_code[depth_code>0]
    depth_code[depth_code > 230] = 230
    mask = np.array(depth_code > 0)
    mask[mask > 0] = 255
    mask = mask.astype(np.uint8)
    depth_cplor = cv2.applyColorMap(depth_code.astype(np.uint8), cv2.COLORMAP_TURBO)  # Turbo
    # depth_cplor = cv2.blur(depth_cplor,(3,3))
    # depth_cplor = cv2.bilateralFilter(depth_cplor,13,75,75)  #13 75 75
    depth_cplor = cv2.blur(depth_cplor, (3, 3))
    depth_cplor = cv2.bitwise_and(depth_cplor, depth_cplor, mask=mask)

    skin = scale_to_255(skin, 0, 255)
    skin = cv2.applyColorMap(skin.astype(np.uint8), cv2.COLORMAP_HOT)

    blood = scale_to_255(blood, 0, 255)
    blood[blood > 220] = 220
    # blood = blood - 50
    # blood[blood<0] = 0
    # blood = scale_to_255(blood, 120, 255)
    blood = cv2.applyColorMap(blood.astype(np.uint8), cv2.COLORMAP_HOT)
    blood = cv2.blur(blood, (3, 3))

    return skin, blood, depth_cplor

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


# def left_click(event, x, y, flags, param):
#     global xp, yp, mouse_move, tracking
#     if event == cv2.EVENT_LBUTTONDOWN:
#         xp, yp = x, y
#         print(xp, yp)
#         ascan = ascan_plot(data, xp, yp)
#         peaks, _ = find_peaks(ascan, height=20, width=3, distance=35)
#         fig = plt.figure(figsize=(20, 10))
#         ax = fig.add_subplot(111)
#         ax.plot(ascan)
#         ax.plot(peaks, ascan[peaks], "x")
#         plt.savefig('ascan.png')
#         ascanimg = cv2.imread("ascan.png")
#         cv2.imshow("A-Scan", ascanimg)
#     elif event == cv2.EVENT_MOUSEMOVE:
#         xp, yp = x, y
#         mouse_move = True





# data, cscan, tof = read_data_from_npy("./result/predict_earmouse_2020.npy")
# cscan = scale_to_255(cscan, 0, 255)
# color = cv2.applyColorMap(cscan.astype(np.uint8), cv2.COLORMAP_HOT)
# cv2.imwrite("tuearmouse.png", color)
# cv2.imshow("a", color)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
# ascan_reconstruct(data, ascan_path, 0, 0.08)
# img = cv2.imread("./data_ascan/hand_tu/2d_ascan_hilbert/50y_120x_ascan.png")
# ascan = img_to_ascan_reconstruct(img)
# ascan = ascan/255
# hilbert = hilbert_scan(ascan)
# plt.plot(ascan)
# plt.plot(hilbert+0.5)
# plt.show()



# data, cscan, tof = read_data_from_npy("./result/predict_tu_100u_60.npy")
# print(data.shape)
# print(np.max(data))

# for i in range(data.shape[0]):
#     # pd.DataFrame(data[i,:,:]).to_csv("./result/bscan_earmouse_dat/bscan_%s.csv" %i)
#     cv2.imwrite("./result/bscan_image/bscan_%s.png" %i, data[i,:,:])
#     print(i)

# mask = data[824:-1, :, 0:370]
# mask2 = data[824:-1, :, 2639:-1]
# print(mask.shape)
#
# data[200:675, :, 300: 670] = mask
# data[0:475, :, 2639:-1] = mask2
#
# data = data.astype(np.uint8)
# np.save("./result/earmous_new.npy", data)

# data, cscan, tof = read_data_from_npy("./result/tufoot_8bit.npy")
# ascan = data[350,:, 500]
#
# fig = plt.figure(figsize=(20, 10))
# ax = fig.add_subplot(111)
# ax.plot(ascan)
# plt.show()



# offset = 0
# skin, blood, tof = cscan_from_predictdata(data, offset)



# cv2.imwrite("tu_%s_skin.png" %offset, skin)
# cv2.imwrite("tu_%s.png" %offset, tof)
# cv2.imwrite("tu_%s_tof.png" %offset, tof)
# cv2.imshow("a", skin)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

ori_data = np.load("./result/foot_ori.npy")
print(ori_data.shape)


data = np.load("./result/foot_segment.npy")
print(data.shape)
# print(np.max(data))
#
data[data>200] = 0
data[data<128] = 0
data[data!=0] = 1

seg = data*ori_data
np.save("./result/foot_blood_segment.npy", seg.astype(np.uint8))
# cscan = []
# num_bscan = data.shape[0]
# for b in range(num_bscan):
#     cscan.append(np.amax(seg[b, :, :], axis=0))
#     print(b)
# cscan = np.array(cscan)
# # cscan[cscan<190] = 190
# # cscan = scale_to_255(cscan, 190, 255)
# cscan = cscan.astype(np.uint8)
# cscan = cv2.applyColorMap(cscan, cv2.COLORMAP_HOT)
# cscan = cv2.blur(cscan, (3,3))
# cv2.imwrite("skin.png", cscan)
# cv2.imshow("a", cscan)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# data = np.zeros((700,1024,1024))
#
# for i in range(700):
#     img = cv2.imread("./aTu/tuFoot/predict/bscan_%s.png" %i)
#     img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#     data[i,:,:] = img
#     print(i)
# data = data.astype(np.uint8)
# np.save("./result/foot_segment.npy", data)





