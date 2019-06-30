import cv2
import dlib

cor = (0,0,255)
#tracking
tracker = dlib.correlation_tracker()
trackingFace = 0
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
                tracker.start_track(img,dlib.rectangle(int(ex+ew/4),ey+5,int(ex+ew/1.5),int(ey+ew/4)))
        cv2.imshow('img',img) 
        k = cv2.waitKey(30) & 0xff
        
    if trackingFace:
        trackingQuality = tracker.update(img)

        if trackingQuality >= 8.75:
            tracked_position =  tracker.get_position()

            t_x = int(tracked_position.left())
            t_y = int(tracked_position.top())
            t_w = int(tracked_position.width())
            t_h = int(tracked_position.height())
            cv2.rectangle(img, (t_x, t_y),(t_x + t_w , t_y + t_h),cor ,2)

        else:
            trackingFace = 0
            
        cv2.imshow('img',img)
        k = cv2.waitKey(30) & 0xff
        
    if k == 27:
        break
        
cap.release() 
cv2.destroyAllWindows() 
