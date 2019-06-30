import cv2
import numpy as np

count = 0
cor = (0,0,255)
#tracking

tracker = cv2.TrackerBoosting_create()
#cv2.illuminationChange
           # cv2.createTracbar
           # cv2.calmanfilter

trackingFace = 0
bbox = (287, 23, 86, 320)

#detecting face
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml') 
cap = cv2.VideoCapture(0)

while 1:
    ret,img = cap.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    if not trackingFace:
        face = face_cascade.detectMultiScale(gray)
        if (len(face)==1):
            
            for (ex,ey,ew,eh) in face:
                roi_gray = gray[ey+5:int(ey+ew/4), int(ex+ew/4):int(ex+ew/1.5)] 
                roi_color = img[ey+5:int(ey+ew/4), int(ex+ew/4):int(ex+ew/1.5)]
                b,g,r = cv2.split(roi_color) 
                cv2.rectangle(img,(int(ex+ew/4),ey+5),(int(ex+ew/1.5),int(ey+ew/4)),cor,2)
                trackingFace = 1
                # Initialize tracker with roi
                ret = tracker.init(gray, roigray)
                bbox = cv2.selectROI(roi_gray, False)
                
        cv2.imshow('img',img) 
        k = cv2.waitKey(30) & 0xff
        
    if trackingFace:
        ret, bbox = tracker.update(gray)
        p1 = (int(bbox[0]), int(bbox[1]))
        p2 = (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3]))
        cv2.rectangle(img, p1, p2, cor, 2)
    
        cv2.imshow('img',img)
        k = cv2.waitKey(30) & 0xff
        
    if k == 27:
        break
        
cap.release() 
