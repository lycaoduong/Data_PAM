import numpy as np
import matplotlib.pyplot as plt
import scipy.fftpack
import cv2
from data_help_function import *
from scipy import signal

# Number of samplepoints
# N = 1500
# # sample spacing
# T = 1.0 / 200
# x = np.linspace(0.0, N*T, N)
# y = np.sin(50.0 * 2.0*np.pi*x) + 0.5*np.sin(80.0 * 2.0*np.pi*x)
# print(y.shape)
# yf = scipy.fftpack.fft(y)
# xf = np.linspace(0.0, 1.0//(2.0*T), N//2)
#
# fig, ax = plt.subplots()
# ax.plot(x,y)
# plt.show()
#
# fig, ax = plt.subplots()
# ax.plot(xf, 2.0/N * np.abs(yf[:N//2]))
# plt.show()

img = cv2.imread("save.png")
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
ascan = img_2_ascan(gray, predict=True)/255
# data, cscan, tof = read_data_from_npy("tu_hand_15khz.npy")
# ascan = ascan_plot(data, 200, 200)
t = np.linspace(0, 7.5, 1500)

fig = plt.figure(figsize=(20, 10))
ax = fig.add_subplot(111)


# ax.plot(t, ascan)
# plt.show()


N = 1500
T = 1.0/200

def butter_highpass(cutoff, fs, order=3):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = signal.butter(order, normal_cutoff, btype='high', analog=False)
    return b, a

def butter_highpass_filter(data, cutoff, fs, order=3):
    b, a = butter_highpass(cutoff, fs, order=order)
    y = signal.filtfilt(b, a, data)
    return y

cutoff = 10
hpf = butter_highpass_filter(ascan, cutoff, 1500)

yf = scipy.fftpack.fft(hpf)
print(yf)
xf = np.linspace(0.0, 1.0//(2.0*T), N//2)
ax.plot(xf, 2.0/N * np.abs(yf[:N//2]))
plt.show()


