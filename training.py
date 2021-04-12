import numpy as np
import os
import matplotlib.pyplot as plt
from model import *
from keras.layers import Input, add, Multiply, Activation
from keras.models import Model
from keras.optimizers import Adam
from keras import backend as K
from datetime import datetime
import cv2
from train_hepler_function import *
from data_help_function import *
from keras.models import load_model
from keras.callbacks import ModelCheckpoint

patch_w = 256
patch_h = 256


file_path = "./aTu/"
xpath = os.path.join(file_path, "origin")
ypath = os.path.join(file_path, "ypath")

data = os.listdir(xpath)
label = os.listdir(ypath)

img_h = 1024
img_w = 1000

x_image = np.zeros((len(data), img_h, img_w))
y_image = np.zeros((len(label), img_h, img_w))


for idx, imname in enumerate(data):
    print(os.path.join(xpath, imname))
    img = cv2.imread(os.path.join(xpath, imname))
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = gray/255
    x_image[idx, :, :] = gray[:, :]

print(x_image.shape)

for idx, imname in enumerate(label):
    print(os.path.join(ypath, imname))
    img = cv2.imread(os.path.join(ypath, imname))
    y_image[idx, :, :] = img[:,:,0]/255

print(y_image.shape)

str_h, str_w = find_stride(img_h, img_w, patch_h, patch_w)

x_train = extract_ordered_overlap(x_image, patch_h, patch_w, str_h, str_w)
y_train = extract_ordered_overlap(y_image, patch_h, patch_w, str_h, str_w)
print("Xtrain: ", x_train.shape, y_train.shape)

x_train = np.expand_dims(x_train, axis=3)
y_train = np.expand_dims(y_train, axis=3)

inpt = Input(shape=(256, 256, 1))
print(inpt.shape)

model = unet_ascan(inpt)
adam = Adam(lr=0.0001)
model.compile(optimizer=adam, loss='binary_crossentropy', metrics=['accuracy'])

checkpoint = ModelCheckpoint("best_model_tuHand.h5", monitor='loss', verbose=1, save_best_only=True, mode='auto', period=1)
# model = load_model('tuFoot.h5')

result = model.fit(x_train, y_train, batch_size=4, epochs=10, validation_split=0.4)
plt.plot(np.arange(len(result.history['accuracy'])), result.history['accuracy'], label='training')
plt.plot(np.arange(len(result.history['val_accuracy'])), result.history['val_accuracy'], label='validation')
plt.title('Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()
model.save('test.h5')

for i in range(4):
    output = model.predict(np.expand_dims(x_train[3+i, :, :, :], axis=0))
    plt.figure()
    plt.subplot(131)
    plt.imshow(np.squeeze(output), cmap="gray")
    plt.subplot(132)
    plt.imshow(y_train[3+i, :, :, 0], cmap="gray")
    plt.subplot(133)
    plt.imshow(x_train[3+i, :, :, 0], cmap="gray")
    plt.show()
