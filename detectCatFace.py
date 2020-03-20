import matplotlib.pyplot as plt
import cv2

#Change the path ! 
cat_face_cascade = cv2.CascadeClassifier('DATA/haarcascades/haarcascade_frontalcatface_extended.xml')

#Show the selected image
cat = cv2.imread('Dataset/photos/mustie0.jpg',0)
plt.imshow(cat,cmap='gray')

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

#call the function and show the result     
result = detectCatFace(cat)
plt.imshow(result,cmap='gray')

#call the extract function and show the result
face = recoverCatFace(cat)
plt.imshow(face,cmap='gray')