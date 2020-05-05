import PIL
import h5py
# import cv2
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras import regularizers
from tensorflow.keras.layers import GlobalAveragePooling2D,Dense
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input, decode_predictions
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import Sequential, load_model

print('Librairies imported...')

def path_to_tensor(img_path):
    """Takes path to single image and returns it as a 4D tensor"""
    # loads RGB image as PIL.Image.Image type
    img = image.load_img(img_path, target_size=(224, 224))
    # convert PIL.Image.Image type to 3D tensor with shape (x, y, 3)
    x = image.img_to_array(img)
    # convert 3D tensor to 4D tensor with shape (1, x, y, 3) and return 4D tensor
    return np.expand_dims(x, axis=0)

def extract_Resnet50(tensor):
    bottleneck_feature = ResNet50(weights='imagenet', include_top=False).predict(preprocess_input(tensor))
    return bottleneck_feature

def Resnet50_predict_breed(img_path):
    """Takes image path, formats it into 4D tensor , returns name of dog corresponding to ResNet50's prediction"""
    # extract bottleneck features
    bottleneck_feature = extract_Resnet50(path_to_tensor(img_path))
    # obtain predicted vector
    predicted_vector = Resnet50_model.predict(bottleneck_feature)
    # return dog breed that is predicted by the model
    return train_targets[np.argmax(predicted_vector)]

train_targets = ["cavalier","unknow"]

print('About to build the model...')

Resnet50_model = Sequential()
Resnet50_model.add(GlobalAveragePooling2D(input_shape=(7,7,2048)))
Resnet50_model.add(Dense(2, activation='softmax'))

Resnet50_model.summary()

print('Loading weights in the empty model...')

Resnet50_model.load_weights('.\logs_cavalier/linear-0/best_Resnet50_model.h5')

img_path = "../dogFlap/cavalier.jpg"
# dog = cv2.imread(img_path,0)
# plt.imshow(dog,cmap='gray')

print('About to predict the breed...')

print('It''s a ... ', Resnet50_predict_breed(img_path), '!!!')