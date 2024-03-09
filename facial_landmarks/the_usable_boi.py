import face_recognition
import cv2
import numpy as np
import os

path = '/home/itsfod/Documents/tests/facial_landmarks/people'

images = []
classNames = []
myList = os.listdir(path)
print('folder contents',myList)

for cl in myList:
    curImg = cv2.imread(f'{path}/{cl}')
    images.append(curImg) 
    classNames.append(os.path.splitext(cl)[0]) # removes the file extension
print('classnames ',classNames)

def findEncodings(images):
    encodedList = [] 
    for img in images:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) # convert to RGB to fit the face_recognition library
        encode = face_recognition.face_encodings(img)[0]
        encodedList.append(encode)
    return encodedList

Known = findEncodings(images)
print(len(Known))

# capture the video from an mp4 file

'''
cap = cv2.VideoCapture('/home/itsfod/Documents/tests/lulu.mp4')

while True:
    success, img = cap.read()
    imgS = cv2.resize(img, (0,0), None, 0.25, 0.25)
    imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)

    facesCurFrame = face_recognition.face_locations(imgS)
    encodesCurFrame = face_recognition.face_encodings(imgS, facesCurFrame)

    for encodeFace, faceLoc in zip(encodesCurFrame, facesCurFrame):
        matches = face_recognition.compare_faces(Known, encodeFace)
        faceDis = face_recognition.face_distance(Known, encodeFace)
        print(faceDis)
        matchIndex = np.argmin(faceDis)
        print(matchIndex)

        if matches[matchIndex]:
            name = classNames[matchIndex].upper()
            print(name)
            y1, x2, y2, x1 = faceLoc
            y1, x2, y2, x1 = y1*4, x2*4, y2*4, x1*4
            cv2.rectangle(img, (x1,y1), (x2,y2), (0,255,0), 2)
            cv2.rectangle(img, (x1,y2-35), (x2,y2), (0,255,0), cv2.FILLED)
            cv2.putText(img, name, (x1+6, y2-6), cv2.FONT_HERSHEY_COMPLEX, 1, (255,255,255), 2)

    cv2.imshow('idk', img)
    cv2.waitKey(1)
'''

# capture the video from the webcam

cap = cv2.VideoCapture(0)

while True:
    success, img = cap.read()
    img1 = cv2.resize(img, (0,0), None, 0.25, 0.25)
    img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)

    facescf = face_recognition.face_locations(img1)
    encodescf = face_recognition.face_encodings(img1, facescf)

    for encodeFace, faceLoc in zip(encodescf, facescf):
        matches = face_recognition.compare_faces(Known, encodeFace)
        faceDis = face_recognition.face_distance(Known, encodeFace)
        print(faceDis)
        matchIndex = np.argmin(faceDis)
        print(matchIndex)

        if matches[matchIndex]:
            name = classNames[matchIndex].upper()
            print(name)
            y1, x2, y2, x1 = faceLoc
            y1, x2, y2, x1 = y1*4, x2*4, y2*4, x1*4
            cv2.rectangle(img, (x1,y1), (x2,y2), (0,255,0), 2)
            cv2.rectangle(img, (x1,y2-35), (x2,y2), (0,255,0), cv2.FILLED)
            cv2.putText(img, name, (x1+6, y2-6), cv2.FONT_HERSHEY_COMPLEX, 1, (255,255,255), 2)

    cv2.imshow('idk', img)
    cv2.waitKey(1)