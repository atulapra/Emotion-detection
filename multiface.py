import cv2
import sys
from em_model import EMR
import numpy as np

EMOTIONS = ['angry', 'disgusted', 'fearful', 'happy', 'sad', 'surprised', 'neutral']

# initialize the cascade
cascade_classifier = cv2.CascadeClassifier('haarcascade_files/haarcascade_frontalface_default.xml')  

# Initialize object of EMR class
network = EMR()
network.build_network()

cap = cv2.VideoCapture(0)
font = cv2.FONT_HERSHEY_SIMPLEX
feelings_faces = []

# append the list with the emoji images
for index, emotion in enumerate(EMOTIONS):
    feelings_faces.append(cv2.imread('./emojis/' + emotion + '.png', -1))

while True:
    # Again find haar cascade to draw bounding box around face
    ret, frame = cap.read()
    facecasc = cv2.CascadeClassifier('haarcascade_files/haarcascade_frontalface_default.xml')
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = facecasc.detectMultiScale(gray, 1.3, 5)
    if not len(faces) > 0:
        # do nothing if no face is detected
        a = 1
    else:
        # draw box around faces
        for face in faces:
            (x,y,w,h) = face
            frame = cv2.rectangle(frame,(x,y-30),(x+w,y+h+10),(255,0,0),2)
            newimg = frame[y:y+h,x:x+w]
            newimg = cv2.resize(newimg, (48,48), interpolation = cv2.INTER_CUBIC) / 255.
            result = network.predict(newimg)
            if result is not None:
                result[0][2] -= 0.15
                result[0][4] -= 0.15
                if result[0][3] > 0.06:
                    result[0][3] += 0.4
                maxindex = np.argmax(result[0])
                font = cv2.FONT_HERSHEY_SIMPLEX
                cv2.putText(frame,EMOTIONS[maxindex],(x+5,y-5), font, 2,(255,255,255),2,cv2.LINE_AA) 

    cv2.imshow('Video', cv2.resize(frame,None,fx=2,fy=2,interpolation = cv2.INTER_CUBIC))
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()