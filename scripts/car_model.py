#! /usr/bin/env python

from cv2 import cv2 as cv
import numpy as np
import math
from os import listdir
from os.path import isfile, join
import ipywidgets as ipywidgets
from ipywidgets import interact
import re
from collections import Counter
import matplotlib.pyplot as plt
from PIL import Image
import seaborn as sn
import pandas as pd
from keras import layers, models, optimizers, backend
from keras.utils import plot_model, to_categorical
from sklearn.metrics import confusion_matrix
import os

path = os.path.dirname(os.path.realpath(__file__)) + "/"
img_dir = path + 'images/read_noread_training/'


#Create list of environment images
env_img_paths = [ join(img_dir, f) for f in listdir(img_dir) if isfile(join(img_dir, f)) ]
pics = np.concatenate([ (cv.imread(path,0))[np.newaxis] for path in env_img_paths ]) #4D array
height = pics[0].shape[0]
width = pics[0].shape[1]
pics = pics.reshape(len(env_img_paths), height, width, 1)

#Collect labels
labels = np.empty(len(listdir(img_dir)))
for n in range(0, len(listdir(img_dir))):
    labels[n] = listdir(img_dir)[n][-5]

#Create X, Y datasets
Y_dataset = to_categorical(labels, num_classes=None, dtype=object) #one hot encoded
X_dataset = pics

VALIDATION_SPLIT = 0.2 #to separate training set from test set
def reset_weights(model):
  session = backend.get_session()
  for layer in model.layers: 
    if hasattr(layer, 'kernel_initializer'):
      layer.kernel.initializer.run(session=session)
conv_model = models.Sequential()
conv_model.add(layers.Conv2D(32, (3, 3), activation='relu', strides = 4, input_shape = (height, width, 1)))
conv_model.add(layers.MaxPooling2D((2, 2)))
conv_model.add(layers.Flatten()) #string zig zag gets unrolled
conv_model.add(layers.Dropout(0.5))
conv_model.add(layers.Dense(512, activation='relu'))
conv_model.add(layers.Dense(2, activation='softmax')) ## of labels
conv_model.summary()

#Train the Model
LEARNING_RATE = 1e-4
conv_model.compile(loss='categorical_crossentropy', optimizer=optimizers.RMSprop(lr=LEARNING_RATE), metrics=['acc'])
history_conv = conv_model.fit(X_dataset, Y_dataset, validation_split=VALIDATION_SPLIT, epochs=20, batch_size=16)

#Plot results
plt.plot(history_conv.history['loss'])
plt.plot(history_conv.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train loss', 'val loss'], loc='upper left')
plt.show()
plt.plot(history_conv.history['acc'])
plt.plot(history_conv.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy (%)')
plt.xlabel('epoch')
plt.legend(['train accuracy', 'val accuracy'], loc='upper left')
plt.show()

#Save Model
#conv_model.save(path + 'car_model.h5')

# #Print incorrectly identified images
# for n in range(0, len(listdir(img_dir))):
#   print(str(n))
#   imgoop = cv.imread(env_img_paths[n], 0)
#   img = imgoop
#   img = img.reshape(height, width, 1)
#   img_aug = np.expand_dims(img, axis=0)
#   if labels[n] != 0:
#     y_predict = conv_model.predict(img_aug)[0]
#     spot = int(np.where(y_predict == np.amax(y_predict))[0])
#     if spot != labels[n] :
#       cv.imshow("Labeled: " + str(labels[n]) + "\tPredicted: " + str(spot), imgoop)
#       cv.waitKey(0)

# #Generate Confusion Matrix
# labels = "01"
# NUM_IMAGES = 60
# y_pred_raw = conv_model.predict(X_dataset[0:NUM_IMAGES]) #one-hot encoded
# y_pred = [np.argmax(y_val) for y_val in y_pred_raw]
# y_gnd  = [np.argmax(y_val) for y_val in Y_dataset[0:NUM_IMAGES]]
# cm = confusion_matrix(y_gnd, y_pred)
# df_cm = pd.DataFrame(cm, index = [i for i in labels], columns = [i for i in labels])
# plt.figure(figsize = (10,7))
# sn.heatmap(df_cm, annot=True, annot_kws={"size": 10})
# plt.show()
