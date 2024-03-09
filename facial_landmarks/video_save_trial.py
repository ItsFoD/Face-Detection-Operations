import cv2
import mediapipe as mp

# Face Mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh()

fourcc = cv2.VideoWriter_fourcc(*'MJPG')  # the codec
out = cv2.VideoWriter('output_video.mp4', fourcc, 60.0, (720, 1280))  

cap = cv2.VideoCapture("lulu.mp4")

while True:
    ret, image = cap.read()
    if ret is not True:
        break
    height, width, _ = image.shape

    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Facial landmarks
    result = face_mesh.process(rgb_image)

    if result.multi_face_landmarks:
        for face_landmarks in result.multi_face_landmarks:
            for i in range(0, 468):
                pt1 = face_landmarks.landmark[i]
                x = int(pt1.x * width)
                y = int(pt1.y * height)
                cv2.circle(image, (x, y), 2, (100, 100, 0), -1)
    out.write(image)

cap.release()
out.release()
cv2.destroyAllWindows()