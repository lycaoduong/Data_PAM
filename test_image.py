import cv2
import numpy as np

img = cv2.imread("./data_ascan/hand_tu/2d_ascan_hilbert/200y_119x_ascan.png")
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


cv2.imshow("a", gray)
cv2.waitKey(0)
cv2.destroyAllWindows()