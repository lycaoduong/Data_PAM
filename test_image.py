import cv2
import numpy as np
from data_help_function import *
import matplotlib.pyplot as plt

img = cv2.imread("./data_ascan/hand_tu/test/200y_294x_ascan_org.png")
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

mask = cv2.imread("./data_ascan/hand_tu/test/predict.png")
mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)

mask_org = np.zeros((256, 1500))
for i in range(1500):
    if np.max(mask[:, i]) >=128:
        mask_org[:, i] = 255

mask_org = mask_org.astype(np.uint8)
mask_plus = mask_org.copy()
mask_plus[mask_org==255] = 0
mask_plus[mask_org==0] = 127
mask_plus[0:128, :] = 0

out = cv2.bitwise_and(img, img, mask = mask_org)
out2 = (out[:,:,0] + mask_plus)/255-0.5
hil = hilbert_scan(out2)

fig = plt.figure(figsize=(20, 10))
ax = fig.add_subplot(111)
ascan = img_2_ascan(out2, predict=False)
ax.plot(ascan)
plt.show()

img = scale_to_255()

