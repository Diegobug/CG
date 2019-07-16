import cv2
import numpy as np
cor = (0,0,255)
#tracking
feature_params = dict( maxCorners = 100,qualityLevel = 0.3,minDistance = 7,blockSize = 7 )
lk_params = dict(winSize  = (15,15),maxLevel = 2,criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
trackingFace = 0
#detecting face
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml') 
cap = cv2.VideoCapture(0)
ret,img = cap.read()
mask = np.zeros_like(img)
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
                
        cv2.imshow('img',img) 
        k = cv2.waitKey(30) & 0xff
        
    if trackingFace:
        p0 = cv2.goodFeaturesToTrack(roi_gray, mask = None, **feature_params)#
        p1, st, err = cv2.calcOpticalFlowPyrLK(roi_gray, gray[roi_gray], p0, None, **lk_params)
        good_new = p1[st==1]
        good_old = p0[st==1]
        
        for i,(new,old) in enumerate(zip(good_new,good_old)):
            a,b = new.ravel()
            c,d = old.ravel()
            mask = cv2.line(mask, (a,b),(c,d), cor, 2)
            frame = cv2.circle(img,(a,b),5,cor,-1)
        img = cv2.add(frame,mask)
        
        roi_gray = gray[roi_gray]
            
        cv2.imshow('img',img)
        k = cv2.waitKey(30) & 0xff
        
    if k == 27:
        break
        
cap.release() 
cv2.destroyAllWindows() 
