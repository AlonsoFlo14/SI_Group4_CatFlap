#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import re
import h5py
import PIL
import cv2
import pickle
import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
from tqdm import tqdm
from tensorflow.keras import regularizers
from tensorflow.keras.layers import GlobalAveragePooling2D,Dense
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input, decode_predictions
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.callbacks import TensorBoard,ModelCheckpoint
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from sklearn.preprocessing import MinMaxScaler,LabelEncoder
from sklearn.model_selection import train_test_split


# In[2]:


img_height = 224
img_width = 224


# In[3]:


def generate_unique_logpath(logdir, raw_run_name):
    i = 0
    while(True):
        run_name = raw_run_name + "-" + str(i)
        log_path = os.path.join(logdir, run_name)
        if not os.path.isdir(log_path):
            return log_path
        i = i + 1


# In[4]:


def path_to_tensor(img_path):
    """Takes path to single image and returns it as a 4D tensor"""
    # loads RGB image as PIL.Image.Image type
    img = image.load_img(img_path, target_size=(img_height, img_width))
    # convert PIL.Image.Image type to 3D tensor with shape (x, y, 3)
    x = image.img_to_array(img)
    # convert 3D tensor to 4D tensor with shape (1, x, y, 3) and return 4D tensor
    return np.expand_dims(x, axis=0)

def paths_to_tensor(img_paths):
    list_of_tensors = [path_to_tensor(img_path) for img_path in tqdm(img_paths)]
    return np.vstack(list_of_tensors)


# In[5]:


def extract_Resnet50(tensor):
    bottleneck_feature = ResNet50(weights='imagenet', include_top=False).predict(preprocess_input(tensor))
    return bottleneck_feature


# In[6]:


cavalier_dir = "../dogFlap/dataset/cavalier/"
unknow_dir = "../dogFlap/dataset/unknow/"
cavalier_list = [cavalier_dir+f for f in os.listdir(cavalier_dir) if re.search('jpg|JPG', f)]
unknow_list = [unknow_dir+f for f in os.listdir(unknow_dir) if re.search('jpg|JPG', f)]
train_list = cavalier_list + unknow_list
train_targets = ["cavalier","unknow"]
train_targets_list = ["cavalier" for i in range(len(cavalier_list))] + ["unknow" for i in range(len(unknow_list))]


# In[7]:


train_tensors = paths_to_tensor(train_list)
del  train_list


# In[8]:


X_train, X_test, y_train, y_test = train_test_split(train_tensors,train_targets_list,test_size=0.20,random_state=42)
del train_tensors, train_targets_list


# In[9]:


y_train = np.array(y_train)
label_encoder_train = LabelEncoder()
y_train = label_encoder_train.fit_transform(y_train)
y_train = to_categorical(y_train)
print(y_train.shape[0])
y_test = np.array(y_test)
label_encoder_test = LabelEncoder()
y_test = label_encoder_test.fit_transform(y_test)
y_test = to_categorical(y_test)
print(y_test.shape[0])


# In[10]:


train_Resnet50 = extract_Resnet50(X_train)
del X_train
test_Resnet50 = extract_Resnet50(X_test)
del X_test


# In[11]:


Resnet50_model = Sequential()
Resnet50_model.add(GlobalAveragePooling2D(input_shape=train_Resnet50.shape[1:]))
Resnet50_model.add(Dense(2, activation='softmax'))

Resnet50_model.summary()


# In[12]:


sgd = SGD(lr=0.01, clipnorm=1, decay=1e-6, momentum = 0.9, nesterov=True)
Resnet50_model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])


# In[13]:


run_name = "linear"
logpath = generate_unique_logpath(".\logs_cavalier", run_name)
print(logpath)
tbcb = TensorBoard(log_dir=logpath)
checkpoint_filepath = os.path.join(logpath,  "best_Resnet50_model.h5")
checkpoint_cb = ModelCheckpoint(checkpoint_filepath, save_best_only=True)


# In[14]:


Resnet50_model.fit(train_Resnet50, y_train,
          batch_size=128,
          epochs=10,
          verbose=1,
          validation_split=0.1,
          callbacks=[tbcb,checkpoint_cb])


# In[15]:


score = Resnet50_model.evaluate(test_Resnet50, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])


# In[17]:


Resnet50_model.load_weights(logpath + '/best_Resnet50_model.h5')


# In[18]:


img_path = "../dogFlap/cavalier.jpg"
dog = cv2.imread(img_path,0)
plt.imshow(dog,cmap='gray')


# In[19]:


def Resnet50_predict_breed(img_path):
    """Takes image path, formats it into 4D tensor , returns name of dog corresponding to ResNet50's prediction"""
    # extract bottleneck features
    bottleneck_feature = extract_Resnet50(path_to_tensor(img_path))
    # obtain predicted vector
    predicted_vector = Resnet50_model.predict(bottleneck_feature)
    # return dog breed that is predicted by the model
    return train_targets[np.argmax(predicted_vector)]


# In[20]:


Resnet50_predict_breed(img_path)


# In[ ]:




