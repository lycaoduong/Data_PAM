import numpy as np
import os
from keras.models import load_model
from datetime import datetime
import cv2
from train_hepler_function import *
from data_help_function import *
import matplotlib.pyplot as plt

file_path = "./data_ascan/hand_tu/"
xpath = os.path.join(file_path, "test")
data = os.listdir(xpath)

width = 1500
height = 256

patch_w = 256
patch_h = 256

def get_dataset_test(x_path):
    data = os.listdir(x_path)
    numfile = len(data)
    x_train = np.zeros([numfile, height, width])
    for idx, imname in enumerate(data):
        print(os.path.join(xpath, imname))
        img = cv2.imread(os.path.join(xpath, imname))
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        x_train[idx, :, :] = gray/255
    return x_train

test = get_dataset_test(xpath)
print(test.shape)

patches = extract_ordered_overlap(test, patch_h, patch_w, 4, 4)
print(patches.shape)


x_train = np.expand_dims(patches, axis=3)
print(x_train.shape)

out_img = np.zeros((312, 256, 256))

model = load_model('ascan_Unet.h5')
for i in range(312):
    predict = model.predict(np.expand_dims(x_train[i, :, :, :], axis = 0), verbose=1)
    predict[predict<0.5] = 0
    predict = np.squeeze(predict, axis=0)*255
    out_img[i] = np.squeeze(predict, axis=2)
    # cv2.imwrite("./data_ascan/hand_tu/predict/%s.png"%i, predict)
img_save = recompose_overlap(out_img, 256, 1500, 4, 4)
img_save = np.squeeze(img_save, axis = 0)
print(img_save)
ascan = img_2_ascan(img_save, predict = True)

fig = plt.figure(figsize=(20, 10))
ax = fig.add_subplot(111)
ax.plot(ascan)
plt.savefig("recompose.png")
plt.show()
cv2.imwrite("./data_ascan/hand_tu/predict/save.png", img_save)


