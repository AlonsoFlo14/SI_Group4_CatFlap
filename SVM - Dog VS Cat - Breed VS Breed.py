#!/usr/bin/env python
# coding: utf-8

# # SVM - Dogs breed detection

# ## Librairies

# In[1]:


#import
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
#from sklearn.cross_validation import train_test_split --> old bibli
from imutils import paths
import numpy as np
import argparse
import imutils
import cv2
import os
import _pickle as cPickle


# ## Pre-processing

# In[2]:


#color histogram
def extract_color_histogram(image, bins=(8, 8, 8)):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv],[0, 1, 2],None,bins,[0, 180, 0, 256, 0, 256])

    if imutils.is_cv2():
        hist = cv2.normalize(hist)
    else:
        cv2.normalize(hist, hist)

    return hist.flatten()


# In[3]:


#get path of images in dataset
imagePaths = list(paths.list_images('./DATA/MyDogs_Breeds/'))


# In[4]:


#initialising
data = []
labels = []


# In[5]:


#Dataset for linear SVM
for (i, imagePath) in enumerate(imagePaths):
    image = cv2.imread(imagePath)
    label = imagePath.split(os.path.sep)[-1].split(".")[0]

    hist = extract_color_histogram(image)
    data.append(hist)
    labels.append(label)

    if i > 0 and i % 45 == 0:
        print("{}/{}".format(i, len(imagePaths)))


# In[6]:


#labels


# In[7]:


#Label encoding
le = LabelEncoder()
labels = le.fit_transform(labels)
labels = le.inverse_transform(labels)
#labels


# In[8]:


le.classes_


# In[9]:


#Splitiing dataset to train and test
trainData, testData, trainLabels, testLabels = train_test_split(np.array(data), labels, test_size=0.25, random_state=42)


# ## SVM part

# In[10]:


#Support Vector Machine
#model = LinearSVC()  #24-36-27-43%
#model.fit(trainData, trainLabels)

#Import svm model
from sklearn import svm
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import LinearSVC
from sklearn.svm import SVC
#model = OneVsRestClassifier(LinearSVC()) #24-36-27-43%
#model = svm.LinearSVC() #24-36-27-43%
#Create a svm Classifier
#model = svm.SVC(kernel='linear',verbose=True) # Linear Kernel better %35-37-45-48
#model = svm.SVC(kernel='poly',verbose=True) # better %50-40-38-41
model = svm.SVC(kernel='rbf') # better %48-41-41-54
#model = svm.SVC(kernel='sigmoid',verbose=True) # better %35-34-41-52
#model = SVC(decision_function_shape='ovo') #better %48-41-41-54
#Train the model using the training sets
model.fit(trainData, trainLabels)


# In[11]:


# Evaluating
predictions = model.predict(testData)
#print(classification_report(testLabels, predictions,target_names=le.classes_))
print(classification_report(testLabels, predictions))


# In[12]:


#Save the model
f = open("model.cpickle", "wb")
f.write(cPickle.dumps(model))
f.close()
#model = cPickle.loads(open('model.cpickle', "rb").read())


# In[13]:


#Predict a picture
singleImage = cv2.imread('./beagleTEST5.jpg')
histt = extract_color_histogram(singleImage)
histt2 = histt.reshape(1, -1)
prediction = model.predict(histt2)
print(prediction)