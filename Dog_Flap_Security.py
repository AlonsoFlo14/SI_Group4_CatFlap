import h5py
import cv2
import time
import serial
import numpy as np
from PIL import Image
from picamera import PiCamera
from picamera.array import PiRGBArray
import matplotlib.pyplot as plt
from tensorflow.keras import regularizers
from tensorflow.keras.layers import GlobalAveragePooling2D,Dense
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input, decode_predictions
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import Sequential, load_model

print('Librairies imported...')

ResNet50_model_ = ResNet50(weights='imagenet')
cat_face_cascade = cv2.CascadeClassifier('haarcascade_frontalcatface_extended.xml')
camera = PiCamera()
camera.resolution = (224, 224)
rawCapture = PiRGBArray(camera, size=(224, 224))
time.sleep(0.1)
ser=serial.Serial("/dev/ttyACM0",9600)
ser.baudrate=9600

def cat_detector(img):
    cat_img = img.copy()
    cat_rects = cat_face_cascade.detectMultiScale(cat_img)
    cat_face = ()
    if cat_rects != ():
        for (x,y,w,h) in cat_rects: 
            cat_face = cat_img[y:y+h,x:x+w]
    if cat_face != ():
        cat=True
    else:
        cat=False
    return cat

def img_to_tensor(img):
    """Takes path to single image and returns it as a 4D tensor"""
    # loads RGB image as PIL.Image.Image type
    # img = Image.resize(img,(224, 224))
    # convert PIL.Image.Image type to 3D tensor with shape (x, y, 3)
    # x = image.img_to_array(img)
    # convert 3D tensor to 4D tensor with shape (1, x, y, 3) and return 4D tensor
    return np.expand_dims(img, axis=0)

def ResNet50_predict_labels(img):
    """Takes image path and returns pretrained ResNet50 model's prediction"""
    # returns prediction vector for image located at img_path
    img = preprocess_input(img_to_tensor(img))
    return np.argmax(ResNet50_model_.predict(img))
    
def dog_detector(img):
    """Returns True if ResNet50's prediction is one of ImageNet's dog categories"""
    prediction = ResNet50_predict_labels(img)
    #see the dictionnary for values
    return ((prediction <= 268) & (prediction >= 151))

def extract_Resnet50(tensor):
    bottleneck_feature = ResNet50(weights='imagenet', include_top=False).predict(preprocess_input(tensor))
    return bottleneck_feature

def Resnet50_predict_breed(img):
    """Takes image path, formats it into 4D tensor , returns name of dog corresponding to ResNet50's prediction"""
    # extract bottleneck features
    bottleneck_feature = extract_Resnet50(img_to_tensor(img))
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

while True:
    
    #print('About to take a capture...')
    
    camera.capture(rawCapture, format="bgr")
    img = rawCapture.array
    
    #print('About to analyze the capture...')
    
    if cat_detector(img)==1:
        print('Cat')
        ser.flush()
        ser.write(str(2).encode('utf-8'))
    elif dog_detector(img)==1:
        print('Dog')
        ans = Resnet50_predict_breed(img)
        print(ans)
        if ans=="cavalier":
            ser.flush()
            ser.write(str(1).encode('utf-8'))
        else:
            ser.flush()
            ser.write(str(2).encode('utf-8'))
    else:
        print('Nothing')
        
    #read_ser=ser.readline()
    #print(read_ser)
    #print('About to display the capture...')
        
    cv2.imshow("Vidéo", img)
    
    rawCapture.truncate(0)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

