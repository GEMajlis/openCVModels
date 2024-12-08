import cv2 as cv
import numpy as np

haar_cascade = cv.CascadeClassifier('haar_face.xml')

capture = cv.VideoCapture(0)

while True:
    is_true, frame = capture.read()
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    faces_react = haar_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=3)
    for (x,y,w,h) in faces_react:
        cv.rectangle(frame, (x,y), (x+w, y+h), (0,255,0), thickness=2)
    cv.imshow('video', frame)

    if cv.waitKey(1) & 0xFF == ord('q'):
        break

capture.release()
cv.destroyAllWindows()

cv.waitKey(0)