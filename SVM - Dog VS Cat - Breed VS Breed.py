#!/usr/bin/env python
# coding: utf-8

# # SVM - Dogs VS Cats detection

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
imagePaths = list(paths.list_images('./DATA/dogs-vs-cats/train/'))


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

    if i > 0 and i % 1000 == 0:
        print("{}/{}".format(i, len(imagePaths)))


# In[6]:


#Label encoding
le = LabelEncoder()
labels = le.fit_transform(labels)
labels = le.inverse_transform(labels)


# In[7]:


#Splitiing dataset to train and test
trainData, testData, trainLabels, testLabels = train_test_split(np.array(data), labels, test_size=0.25, random_state=42)


# ## SVM part

# In[8]:


#Support Vector Machine
model = LinearSVC()
model.fit(trainData, trainLabels)


# In[9]:


#Evaluating
predictions = model.predict(testData)
#print(classification_report(testLabels, predictions,target_names=le.classes_))
print(classification_report(testLabels, predictions))


# In[10]:


#Save the model
f = open("model.cpickle", "wb")
f.write(cPickle.dumps(model))
f.close()
#model = cPickle.loads(open('model.cpickle', "rb").read())


# In[11]:


#Predict a picture
singleImage = cv2.imread('./DATA/dogs-vs-cats/test1/test1/2.jpg')
histt = extract_color_histogram(singleImage)
histt2 = histt.reshape(1, -1)
prediction = model.predict(histt2)
print(prediction)