import cv2

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

img = cv2.imread("ORL_Faces/s5/14.png")

faces = face_cascade.detectMultiScale(img, 1.1, 4)
for (x,y,w,h) in faces:
    cv2.rectangle(img, (x,y), (x+w,y+h), (0,255,255),1)

cv2.imshow('rez',img)
cv2.waitKey()
