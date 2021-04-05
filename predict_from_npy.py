import numpy as np
import os
from keras.models import load_model
from datetime import datetime
import cv2
from train_hepler_function import *
from data_help_function import *
import matplotlib.pyplot as plt

width = 1536
height = 256

patch_w = 256
patch_h = 256

num_bscan = 700
num_ascan = 1024
depth = 1280

min = 0
max = 255

file_per_predict = 1


data, cscan, tof = read_data_from_npy("./result/tuFoot.npy")
print(data.shape)

pre_pare_data = np.zeros((num_bscan, depth, num_ascan))
predict_data  = np.zeros((num_bscan, depth, num_ascan))

x_train = np.zeros([file_per_predict, height, depth])

pre_pare_data[:,0:1199,0:999]= data[:,0:-1,0:-1]
print(pre_pare_data.shape)


model = load_model('ascan_Unet.h5')

for b in range(num_bscan):
    print("B-scan", b)
    for a in range(num_ascan):
        print("A-scan", a)
        ascan = pre_pare_data[b,:,a]
        ascan_img = ascan_2_img(ascan, min, max)/255
        x_train[0,:,:] = ascan_img
        patches = extract_ordered_overlap(x_train, patch_h, patch_w, 256, 256)
        # print(patches.shape)
        predict = model.predict(patches, verbose=1)
        predict[predict < 0.5] = 0
        predict = np.squeeze(predict, axis=3) * 255
        # print(predict.shape)
        img_predict = recompose_overlap(predict, 256, depth, 256, 256)
        img_predict = np.squeeze(img_predict, axis=0)
        ascan_predict = img_2_ascan(img_predict, predict=True)
        predict_data[b, :, a] = ascan_predict
    np.save("./result/foot_bscan/bscan_%s"%b, predict_data[b,:,:])

np.save('./result/predict_foot_2020.npy', predict_data)



