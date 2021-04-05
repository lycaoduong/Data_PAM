import cv2
import numpy as np
from data_help_function import *
import matplotlib.pyplot as plt


img = cv2.imread("earmouse.JPEG")

imgnew = img[:, 709:-1, :]

cv2.imwrite("newear.png", imgnew)
cv2.imshow("A", imgnew)
cv2.waitKey(0)
cv2.destroyAllWindows()



