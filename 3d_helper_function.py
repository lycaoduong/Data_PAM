from nptdms import TdmsFile
import numpy as np
import os
import matplotlib
from pyevtk.hl import gridToVTK
from scipy.signal import hilbert
import cv2
import itk


# data = np.load("./result/predict_tu_100u_60.npy")
data = np.load("./result/foot_skin_segment.npy")

def numpy2nrrd(npdata):
    image = itk.GetImageFromArray(npdata.astype(np.uint8))
    return  image

image = numpy2nrrd(data)
itk.imwrite(image, "./result/foot_skin.nrrd")

print("Finish")