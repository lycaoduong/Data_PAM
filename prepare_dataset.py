import os
import matplotlib.pyplot as plt
from model import *
from keras.layers import Input, add, Multiply, Activation
from keras.models import Model
from keras.optimizers import Adam
import cv2
from train_hepler_function import *

file_path = "./data_ascan/hand_tu/"
xpath = os.path.join(file_path, "2d_ascan_hilbert")
ypath = os.path.join(file_path, "label")

width = 1500
height = 256

patch_w = 256
patch_h = 256

total_img = 625
patch_per_img = 3
total_patch = total_img*patch_per_img

# data = os.listdir(xpath)
# label = os.listdir(ypath)

# Make sure data and label with the same name
def get_datasets(x_path, y_path):
    data = os.listdir(x_path)
    label = os.listdir(y_path)
    numfile = len(data)
    x_train = np.zeros([numfile, height, width])
    y_train = np.zeros([numfile, height, width])
    for idx, imname in enumerate(data):
        print(os.path.join(xpath, imname))
        img = cv2.imread(os.path.join(xpath, imname))
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        x_train[idx, :, :] = gray/255
        mask = cv2.imread(os.path.join(y_path, imname))
        y_train[idx, :, :] = mask[:,:,0]/255
    return x_train, y_train

imgs, labels = get_datasets(xpath, ypath)
print(imgs.shape)
print(labels.shape)

pathes_data, patches_mask = extract_random(imgs,labels, patch_h,patch_w, total_patch)
print(pathes_data.shape)
print(patches_mask.shape)