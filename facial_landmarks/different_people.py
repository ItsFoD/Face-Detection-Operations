import face_recognition
import cv2
import numpy as np

img = face_recognition.load_image_file('/home/itsfod/Documents/tests/facial_landmarks/images/ali1.jpeg')
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
imgTest = face_recognition.load_image_file('/home/itsfod/Documents/tests/facial_landmarks/images/hema1.jpeg')
imgTest = cv2.cvtColor(imgTest, cv2.COLOR_BGR2RGB)

# Resizing the test image to the same size as the original image ###### they have to be both of the same type
imgTest = cv2.resize(imgTest, (img.shape[1], img.shape[0]))

faceLoc = face_recognition.face_locations(img)[0]
encode = face_recognition.face_encodings(img)[0]
print('top, right, bottom, left', faceLoc)
cv2.rectangle(img, (faceLoc[3], faceLoc[0]), (faceLoc[1], faceLoc[2]), (255, 0, 255), 2)  # top, right, bottom, left, color, thickness

faceLocTest = face_recognition.face_locations(imgTest)[0]
encodeTest = face_recognition.face_encodings(imgTest)[0]
cv2.rectangle(imgTest, (faceLocTest[3], faceLocTest[0]), (faceLocTest[1], faceLocTest[2]), (255, 0, 255), 2)

results = face_recognition.compare_faces([encode], encodeTest)
faceDis = face_recognition.face_distance([encode], encodeTest) # face distance is a measure of how similar the faces are
print(results, faceDis)
cv2.putText(imgTest, f'{results} {round(faceDis[0], 2)}', (50, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 2)

# Concatenate the two images horizontally
combined_img = np.hstack((img, imgTest))

cv2.imshow('Combined Images', combined_img)
cv2.waitKey(0)
cv2.destroyAllWindows()
