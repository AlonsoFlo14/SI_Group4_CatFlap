#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import re
import numpy as np 
from tqdm import tqdm
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input, decode_predictions
from tensorflow.keras.preprocessing import image                  


# In[2]:


ResNet50_model_ = ResNet50(weights='imagenet')


# In[3]:


def path_to_tensor(img_path):
    """Takes path to single image and returns it as a 4D tensor"""
    # loads RGB image as PIL.Image.Image type
    img = image.load_img(img_path, target_size=(224, 224))
    # convert PIL.Image.Image type to 3D tensor with shape (224, 224, 3)
    x = image.img_to_array(img)
    # convert 3D tensor to 4D tensor with shape (1, 224, 224, 3) and return 4D tensor
    return np.expand_dims(x, axis=0)

def paths_to_tensor(img_paths):
    """Takes list of image paths and returns array of 4D image arrays"""
    list_of_tensors = [path_to_tensor(img_path) for img_path in tqdm(img_paths)]
    return np.vstack(list_of_tensors)


# In[4]:


def ResNet50_predict_labels(img_path):
    """Takes image path and returns pretrained ResNet50 model's prediction"""
    # returns prediction vector for image located at img_path
    img = preprocess_input(path_to_tensor(img_path))
    return np.argmax(ResNet50_model_.predict(img))


# In[5]:


### returns "True" if a dog is detected in the image stored at img_path
def dog_detector(img_path):
    """Returns True if ResNet50's prediction is one of ImageNet's dog categories"""
    prediction = ResNet50_predict_labels(img_path)
    #see the dictionnary for values
    return ((prediction <= 268) & (prediction >= 151))


# In[ ]:




