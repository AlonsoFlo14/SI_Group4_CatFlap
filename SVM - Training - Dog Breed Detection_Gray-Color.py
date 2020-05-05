#!/usr/bin/env python
# coding: utf-8

# # SVM - Training - Dog Breed Detection

# ## Librairies

# In[1]:


import os
import re
import PIL
import cv2
import numpy as np   
import pandas as pd
from tqdm import tqdm
from sklearn.preprocessing import MinMaxScaler,LabelEncoder
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical


# ## Variables

# In[2]:


normalize=1
#num_classes = 10
img_height = 125
img_width = 125


# ## Pre-processing

# In[3]:


#Image processing
def path_to_tensor(img_path):
    color = cv2.imread(img_path, 1)
    color = cv2.cvtColor(color, cv2.COLOR_RGB2BGR)
    resized_color = cv2.resize(color,(img_height,img_width))
    x = np.array(resized_color)
    return np.expand_dims(x, axis=0)
#Apply image processing to each image, path
def paths_to_tensor(img_paths):
    list_of_tensors = [path_to_tensor(img_path) for img_path in tqdm(img_paths)]
    return np.vstack(list_of_tensors)


# In[4]:


#Data recovery
train_dir = './DATA/train/'
test_dir = './DATA/test/'
list_train = [train_dir+f for f in os.listdir(train_dir) if re.search('jpg|JPG', f)]
list_test = [test_dir+f for f in os.listdir(test_dir) if re.search('jpg|JPG', f)]
print(list_train[0:4])
print()
print(list_test[0:4])


# In[5]:


data = pd.read_csv('DATA/labels.csv')
data.head(5)


# In[6]:


#Labels recovery
train_labels = data.iloc[:,1].values
train_labels.shape[0]


# In[7]:


train_labels


# In[8]:


#Grouping and counting by breed
dog_names = data.groupby("breed").count()
dog_names = dog_names.rename(columns = {"id" : "count"})
dog_names = dog_names.sort_values("count", ascending=False)
dog_names.head()


# In[9]:


#Shape displays
print(len(list_train))
print(len(list_test))
print(len(dog_names))
print('train_labels.shape',train_labels.shape)


# In[10]:


#Converting each image into a tensor
train_tensors = paths_to_tensor(list_train)


# In[11]:


#Display train_tensors
#train_tensors


# In[12]:


##Splitiing dataset to train and test
X_train, X_test, y_train, y_test = train_test_split(train_tensors,train_labels,test_size=0.30,random_state=42)


# In[13]:


y_train.shape


# In[14]:


#Data reshaping

#num_train  = X_train.shape[0]
#num_test   = X_test.shape[0]

##img_height = X_train.shape[1]
#img_width  = X_train.shape[2]
#X_train = X_train.reshape(num_train, img_width , img_height,1)
#X_test  = X_test.reshape(num_test, img_width , img_height,1)

nb_train = X_train.shape[0]
X_train = X_train.reshape(nb_train,125*125*3)
nb_test = X_test.shape[0]
X_test = X_test.reshape(nb_test,125*125*3)

y_train = np.array(y_train)
label_encoder_train = LabelEncoder()
y_train = label_encoder_train.fit_transform(y_train)
#y_train = to_categorical(y_train)
print(y_train.shape[0])
y_test = np.array(y_test)
label_encoder_test = LabelEncoder()
y_test = label_encoder_test.fit_transform(y_test)
#y_test = to_categorical(y_test)
print(y_test.shape[0])


# In[15]:


X_train.shape[0]


# ## Normalization

# In[16]:


if normalize :
    scaler = MinMaxScaler()
    scaler.fit(X_train)
    X_train=scaler.transform(X_train)
    X_test=scaler.transform(X_test)


# In[17]:


X_train.shape


# In[18]:


y_train.shape


# ## SVM part

# In[19]:


#Import svm model
from sklearn import svm


#from sklearn.multiclass import OneVsRestClassifier
#from sklearn.svm import LinearSVC
#clf = OneVsRestClassifier(LinearSVC())
#clf = svm.LinearSVC()
#Create a svm Classifier
clf = svm.SVC(kernel='linear',verbose=True) # Linear Kernel


# In[20]:


#Train the model using the training sets
clf.fit(X_train, y_train)


# In[21]:


#Predict the response for test dataset
y_pred = clf.predict(X_test)


# In[22]:


#Import scikit-learn metrics module for accuracy calculation
from sklearn import metrics

# Model Accuracy: how often is the classifier correct?
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))


# In[23]:


# Model Precision: what percentage of positive tuples are labeled as such?
print("Precision:",metrics.precision_score(y_test, y_pred,average='weighted'))


# In[24]:


# Model Recall: what percentage of positive tuples are labelled as such?
print("Recall:",metrics.recall_score(y_test, y_pred,average='weighted'))