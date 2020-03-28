import numpy as np
import cv2

cap = cv2.VideoCapture(0)
if (cap.isOpened() is True):
 print('Video camera opened')

while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()

    # Our operations on the frame come here
    imgcolor = cv2.cvtColor(frame, cv2.COLOR_RGB2RGBA)
    # Face detection
    # Load the cascade
    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

    # Convert into grayscale
    gray = cv2.cvtColor(imgcolor, cv2.COLOR_BGR2GRAY)
    # Detect faces
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)
    # Draw rectangle around the faces
    for (x, y, w, h) in faces:
        cv2.rectangle(imgcolor, (x, y), (x+w, y+h), (255, 0, 0), 2)
    # Display the output
    #cv2.imshow('img', img)

    # Display the resulting frame
    cv2.imshow('frame',imgcolor)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        print('Waiting')
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()