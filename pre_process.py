import os
import cv2

file_path = "./aTu/tuFoot_train/"
origin = os.path.join(file_path, "images")
bloodpath = os.path.join(file_path, "blood")
save_path = "./aTu/tuFoot_train/label/"


data = os.listdir(origin)

for idx, imname in enumerate(data):
    print(os.path.join(origin, imname))
    org = cv2.imread(os.path.join(origin, imname))
    org = cv2.blur(org, (3, 3))
    img, thresh = cv2.threshold(org, 100, 255, cv2.THRESH_BINARY)
    thresh = thresh[:,:,0]

    blood = cv2.imread(os.path.join(bloodpath, imname))
    blood = cv2.blur(blood, (3, 3))
    blood = blood[:,:,0]
    blood[blood==255] = 127
    x, mask = cv2.threshold(blood, 50, 127, cv2.THRESH_BINARY_INV)


    skin = cv2.bitwise_and(thresh, thresh, mask=mask)

    merge = skin + blood

    cv2.imwrite(save_path + imname, merge)