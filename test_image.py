import cv2
import numpy as np
from data_help_function import *
import matplotlib.pyplot as plt

#
# img = cv2.imread("earmouse.JPEG")
#
# imgnew = img[:, 709:-1, :]
#
# cv2.imwrite("newear.png", imgnew)
# cv2.imshow("A", imgnew)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

import numpy as np
import os
from keras.models import load_model
from datetime import datetime
import cv2

a = datetime.now()
file_path = "./data/train/"
xpath = os.path.join(file_path, "resize_images")
data = os.listdir(xpath)

x_train = np.zeros((len(data), 1024, 1024))
org = np.zeros((len(data),1024,1024,3))





