import os
import matplotlib.pyplot as plt
from model import *
from keras.layers import Input, add, Multiply, Activation
from keras.models import Model
from keras.optimizers import Adam
import cv2
from train_hepler_function import *

file_path = "./data_ascan/hand_tu/"
# xpath = os.path.join(file_path, "2d_ascan_hilbert")
# ypath = os.path.join(file_path, "label")
xpath = os.path.join(file_path, "img")
ypath = os.path.join(file_path, "lb")

width = 1500
height = 256

patch_w = 256
patch_h = 256

total_img = 34
patch_per_img = 3
total_patch = total_img*patch_per_img

# data = os.listdir(xpath)
# label = os.listdir(ypath)

# Make sure data and label with the same name
def get_datasets(x_path, y_path):
    data = os.listdir(x_path)
    label = os.listdir(y_path)
    numfile = len(data)
    x_train = np.zeros([numfile, height, width])
    y_train = np.zeros([numfile, height, width])
    for idx, imname in enumerate(data):
        print(os.path.join(xpath, imname))
        img = cv2.imread(os.path.join(xpath, imname))
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        x_train[idx, :, :] = gray/255
        mask = cv2.imread(os.path.join(y_path, imname))
        y_train[idx, :, :] = mask[:,:,0]/255
    return x_train, y_train

imgs, labels = get_datasets(xpath, ypath)
print(imgs.shape)
print(labels.shape)


pathes_data, patches_mask = extract_random(imgs,labels, patch_h,patch_w, total_patch)
print(pathes_data.shape)
print(patches_mask.shape)

pathes_data = np.expand_dims(pathes_data, axis=3)
patches_mask = np.expand_dims(patches_mask, axis=3)

inpt = Input(shape=(256, 256, 1))
model = unet_ascan(inpt)
adam = Adam(lr=0.0001)
model.compile(optimizer=adam, loss='binary_crossentropy', metrics=['accuracy'])
result = model.fit(pathes_data, patches_mask, batch_size=4, epochs=50, validation_split=0.2)
plt.plot(np.arange(len(result.history['accuracy'])), result.history['accuracy'], label='training')
plt.plot(np.arange(len(result.history['val_accuracy'])), result.history['val_accuracy'], label='validation')
plt.title('Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()
model.save('ascan_Unet.h5')

for i in range(4):
    output = model.predict(np.expand_dims(pathes_data[3+i, :, :, :], axis=0))
    plt.figure()
    plt.subplot(131)
    plt.imshow(np.squeeze(output), cmap="gray")
    plt.subplot(132)
    plt.imshow(patches_mask[3+i, :, :, 0], cmap="gray")
    plt.subplot(133)
    plt.imshow(pathes_data[3+i, :, :, 0], cmap="gray")
    plt.show()