# coding: utf-8

import cv2,os

os.chdir('E:\\Prog\\git\\face-recognition\\linuxhint')
if not os.path.exists("dataset"):
    os.makedirs("dataset")

location = "C:/Users/lesec/AppData/Local/Programs/Python/Python36-32/Lib/site-packages/cv2/data/"
face_detector = cv2.CascadeClassifier(location + 'haarcascade_frontalface_default.xml')

vid_cam = cv2.VideoCapture(0)
face_id = 1
count = 0

while(vid_cam.isOpened()):
    ret, image_frame = vid_cam.read()
    gray = cv2.cvtColor(image_frame, cv2.COLOR_BGR2GRAY)
    faces = face_detector.detectMultiScale(gray, 1.3, 5)
    
    for (x,y,w,h) in faces:
        cv2.rectangle(image_frame, (x,y), (x+w,y+h), (255,0,0), 2)
        count += 1
        cv2.imwrite("dataset/User-" + str(face_id) + '-' + str(count) + ".jpg", gray[y:y+h,x:x+w])
        cv2.imshow('frame', image_frame)
    
    if cv2.waitKey(100) & 0xFF == ord('q'):
        break
    elif count > 100:
        break

vid_cam.release()
cv2.destroyAllWindows()