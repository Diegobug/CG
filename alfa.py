import cv2
import numpy as np

cor = (0,0,255)
file = open("testfile.txt", "w") 
t = 0.25
#detecting face
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml') 
cap = cv2.VideoCapture(1)

while 1:
    ret,img = cap.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    cv2.equalizeHist(gray)
    face = face_cascade.detectMultiScale(gray)
    if (len(face)==1):
            
        for (ex,ey,ew,eh) in face:
            roi_gray = gray[ey+6:int(ey+ew/4.2), int(ex+ew/4.2):int(ex+ew/1.6)] 
            roi_color = img[ey+6:int(ey+ew/4.2), int(ex+ew/4.2):int(ex+ew/1.6)]
            b,g,r = cv2.split(roi_color) 
            file.write(str(t)+":"+str(int(np.mean(g)))+"\n") 
            t = t+0.25
            cv2.rectangle(img,(int(ex+ew/4.2),ey+6),(int(ex+ew/1.6),int(ey+ew/4.2)),cor,2)
         
    cv2.imshow('img',img)
    k = cv2.waitKey(250) & 0xff
        
    if k == 27:
        break
file.close()       
cap.release() 
cv2.destroyAllWindows() 
