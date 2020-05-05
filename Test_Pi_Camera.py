# import the necessary packages
from picamera.array import PiRGBArray
from picamera import PiCamera
import time
import cv2

# initialize the camera and grab a reference to the raw camera capture
camera = PiCamera()
camera.resolution = (640, 480)
rawCapture = PiRGBArray(camera, size=(640, 480))

# allow the camera to warmup
time.sleep(0.1)

# grab an image from the camera
camera.capture(rawCapture, format="bgr")
print(type(rawCapture))
image = rawCapture.array
print(type(image))
print(image.shape)

# display the image on screen and wait for a keypress
cv2.imshow("Image", image)
cv2.waitKey(0)