import cv2
import matplotlib.pyplot as plt
import numpy as np

#Change the path !
catFaceCascade = cv2.CascadeClassifier('DATA/haarcascades/haarcascade_frontalcatface_extended.xml')

#Cat face detection function
def detectCatFace(img):
    cat_img = img.copy()
    cat_rects = cat_face_cascade.detectMultiScale(cat_img)
    for (x,y,w,h) in cat_rects: 
        cv2.rectangle(cat_img, (x,y), (x+w,y+h), (255,255,255), 10) 
    return cat_img

#function to extract cat's face    
def recoverCatFace(img):
    cat_img = img.copy()
    cat_rects = cat_face_cascade.detectMultiScale(cat_img)
    cat_face = ()
    if cat_rects != ():
        for (x,y,w,h) in cat_rects: 
            cat_face = cat_img[y:y+h,x:x+w]
    return cat_face
    
cap = cv2.VideoCapture(0)

width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

#there are maybe some images in your file 
#change the image number if you already use this code
#if you don't want to rewrite on your images
imgNumb = 107

while True:
    
    # Capture frame-by-frame
    ret, frame = cap.read()

    # Our operations on the frame come here
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    detected_cat = detectCatFace(gray)
    
    catFace = recoverCatFace(gray)
    #execute the "imwrite" only if there is a cat face
    if catFace != ():
        path = './Dataset/Mustie/mustieFace' + str(imgNumb) + '.jpg'
        cv2.imwrite(path,catFace)
        imgNumb += 1
    
    # Display the resulting frame
    cv2.imshow('Cat Detection on Video', detected_cat)
    
    # This command let's us quit with the "q" button on a keyboard.
    # Simply pressing X on the window won't work!
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
        
# When everything done, release the capture and destroy the windows
cap.release()
cv2.destroyAllWindows()
