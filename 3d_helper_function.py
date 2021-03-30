from nptdms import TdmsFile
import numpy as np
import os
import matplotlib
from pyevtk.hl import gridToVTK
from scipy.signal import hilbert
import cv2
import itk

# data = np.load("./result/predict_tu_100u_60.npy")
Data = np.zeros((800,1024,1024))

for i in range(800):
    print(i)
    img = cv2.imread("./aTu/bscan_predict/bscan_%s.png" %i)
    img = img[:,:,0]
    Data[i,:,:] = img

image = itk.GetImageFromArray(Data.astype(np.uint8))
itk.imwrite(image, "./result/segmentation.nrrd")

print("Finish")