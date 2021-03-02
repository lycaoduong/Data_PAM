import cv2
import numpy as np

img = cv2.imread("subtof.png")
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#
#
# blur = cv2.blur(img,(5,5))
ret,thresh1 = cv2.threshold(gray,128,255,cv2.THRESH_BINARY_INV)
blur = cv2.blur(thresh1,(5,5))

ret,thresh2 = cv2.threshold(blur,100,255,cv2.THRESH_TOZERO)

sub_tof_c = cv2.applyColorMap(thresh2.astype(np.uint8), cv2.COLORMAP_HOT)
# #
# #
# org = cv2.imread("sublayer.png")
# result = cv2.bitwise_and(org, org, mask=thresh1)
# cv2.imwrite("sub_mask_1.png", result)


cv2.imshow("a", thresh2)
cv2.waitKey(0)
cv2.destroyAllWindows()