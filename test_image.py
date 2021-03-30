import cv2
import numpy as np
from data_help_function import *
import matplotlib.pyplot as plt


img = cv2.imread("./aTu/xpath/bscan_374.png")

img, thresh = cv2.threshold(img, 100, 255, cv2.THRESH_TOZERO)
blur = cv2.blur(thresh, (3,3))
# img2 = cv2.imread("tu_125.png")
# merge = np.asarray(img)
#
# merge[:,0:500, :] = img2[:, 0:500, :]
# mask = cv2.imread("./skin_noisefilter.png")
# alpha = 0.8
# beta = 1-alpha
# dst = cv2.addWeighted(img, alpha, mask, beta, 0)
# cv2.imwrite("skin_blood.png", dst)
cv2.imshow("merge", blur)
cv2.imwrite("Tu_10_60_blood.png", blur)
cv2.waitKey(0)
cv2.destroyAllWindows()

# data, cscan, tof = read_data_from_npy("./DATA/earmice_2020.npy")
#
# bscan = data.shape[0]
# for i in range(bscan):
#     print(i)
#     bscan_img = data[i,:,:]
#     # bscan_img = cv2.resize(bscan_img, (3010, 1800))
#     bscan_img = scale_to_255(bscan_img, 0, 100)
#     cv2.imwrite("./aTu/bscan_earmouse/bscan_%s.png" %i, bscan_img)


