#!/usr/bin/env python
# coding: utf-8

# # Dog detector and Transfert Learning

# ## Librairies

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


# ## Dog Detector

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


# ## Transfert Learning

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


train_dir = '../dogFlap/dog-breed-identification/train/'
list_train = [train_dir+f for f in os.listdir(train_dir) if re.search('jpg|JPG', f)]


# In[7]:


data = pd.read_csv('../dogFlap/dog-breed-identification/labels.csv')
train_labels = data.iloc[:,1].values


# In[81]:


dog_names = data.groupby("breed").count()
len(dog_names)
#dog_names = dog_names.rename(columns = {"id" : "count"})
#dog_names = dog_names.sort_values("count", ascending=False)
dog_names = dog_names.index
dog_names


# In[9]:


train_tensors = paths_to_tensor(list_train)
del  list_train
#pickle_out = open("train_tensors.pickle","wb")
#pickle.dump(paths_to_tensor(list_train).astype('float32')/255, pickle_out)
#pickle_out.close()


# In[10]:


#pickle_in = open("train_tensors.pickle","rb")
#train = pickle.load(pickle_in)
X_train, X_test, y_train, y_test = train_test_split(train_tensors,train_labels,test_size=0.20,random_state=42)
del train_tensors, train_labels
#pickle_out = open("train_tensors.pickle","wb")
#pickle.dump(train_tensors, pickle_out)
#pickle_out.close()


# In[11]:


print('(',X_train.shape[0],',',X_train.shape[1],',',X_train.shape[2],',',X_train.shape[3],')')
print(type(X_train))
num_train  = X_train.shape[0]
num_test   = X_test.shape[0]

img_height = X_train.shape[1]
img_width  = X_train.shape[2]
X_train = X_train.reshape(num_train, img_width , img_height,3)
X_test  = X_test.reshape(num_test, img_width , img_height,3)
print('(',X_train.shape[0],',',X_train.shape[1],',',X_train.shape[2],',',X_train.shape[3],')')
print(type(X_train))

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


# In[12]:


train_Resnet50 = extract_Resnet50(X_train)
del X_train
test_Resnet50 = extract_Resnet50(X_test)
del X_test


# In[13]:


Resnet50_model = Sequential()
Resnet50_model.add(GlobalAveragePooling2D(input_shape=train_Resnet50.shape[1:]))
Resnet50_model.add(Dense(len(dog_names), activation='softmax'))

Resnet50_model.summary()


# In[14]:


sgd = SGD(lr=0.01, clipnorm=1, decay=1e-6, momentum = 0.9, nesterov=True)
Resnet50_model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])


# In[15]:


run_name = "linear"
logpath = generate_unique_logpath(".\logs_linear", run_name)
print(logpath)
tbcb = TensorBoard(log_dir=logpath)
checkpoint_filepath = os.path.join(logpath,  "best_Resnet50_model.h5")
checkpoint_cb = ModelCheckpoint(checkpoint_filepath, save_best_only=True)


# In[16]:


Resnet50_model.fit(train_Resnet50, y_train,
          batch_size=128,
          epochs=20,
          verbose=1,
          validation_split=0.1,
          callbacks=[tbcb,checkpoint_cb])


# In[17]:


score = Resnet50_model.evaluate(test_Resnet50, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])


# In[20]:


Resnet50_model.load_weights('../dogFlap/logs_linear/linear-9/best_Resnet50_model.h5')


# In[21]:


test_dir = '../dogFlap/dog-breed-identification/test/'
list_test = [test_dir+f for f in os.listdir(test_dir) if re.search('jpg|JPG', f)]


# In[131]:


img_path = "../dogFlap/gaspard.jpg"
dog = cv2.imread(img_path,0)
plt.imshow(dog,cmap='gray')


# In[132]:


def Resnet50_predict_breed(img_path):
    """Takes image path, formats it into 4D tensor , returns name of dog corresponding to ResNet50's prediction"""
    # extract bottleneck features
    bottleneck_feature = extract_Resnet50(path_to_tensor(img_path))
    # obtain predicted vector
    predicted_vector = Resnet50_model.predict(bottleneck_feature)
    # return dog breed that is predicted by the model
    return dog_names[np.argmax(predicted_vector)]


# In[133]:


Resnet50_predict_breed(img_path)


# In[ ]:




