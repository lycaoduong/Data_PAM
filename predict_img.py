import numpy as np
import os
from keras.models import load_model
from datetime import datetime
import cv2
from train_hepler_function import *
from data_help_function import *
import matplotlib.pyplot as plt

model = load_model('tuFoot_new.h5')

patch_w = 256
patch_h = 256

file_path = "./aTu/"
xpath = os.path.join(file_path, "orgin")
ypath = os.path.join(file_path, "ypath")

data = os.listdir(xpath)
label = os.listdir(ypath)


x_image = np.zeros((1, 1024, 1024))
# Data, cscan, tof = read_data_from_npy("./result/tuFoot.npy")
# Data[Data<15] = 0
numBScan= 700

for i in range(numBScan):
    print(i)
    img = cv2.imread("./aTu/tuFoot/bscan_%s.png" %i)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = gray/255
    gray = img/255
    x_image[0, :, 0:1000] = gray[0:1024, :, 0]
    patches = extract_ordered_overlap(x_image, patch_h, patch_w, 256, 256)
    predict = model.predict(patches, verbose=1)
    predict = np.squeeze(predict, axis=3)
    img_predict = recompose_overlap(predict, 1024, 1024, 256, 256)
    img_predict = np.squeeze(img_predict, axis=0)*255
    cv2.imwrite("./aTu/tuFoot/predict/bscan_%s.png" %i, img_predict)

# plt.figure(figsize = (12,10))
# plt.imshow(img_predict)
# plt.colorbar()
# plt.savefig("predict265.png")
# plt.show()

