import cv2
import os
path_in = "./data_ascan/hand_tu/2d_ascan_hilbert/"
path_out = "./data_ascan/hand_tu/label/"


# img = cv2.imread(path_in+file)
# gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# a, img = cv2.threshold(gray, 58, 255, cv2.THRESH_BINARY)
# cv2.imshow("a", img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

def mask_create(input, output):
    for file in os.listdir(input):
        img = cv2.imread(path_in + file)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img, thresh = cv2.threshold(gray, 60, 255, cv2.THRESH_BINARY)
        cv2.imwrite(output + file, thresh)


mask_create(path_in, path_out)
