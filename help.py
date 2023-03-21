import cv2
from deepface import DeepFace

face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

web_cam = cv2.VideoCapture(0)

while True:
    _, frame = web_cam.read()

    to_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(to_gray, 1.1, 4)

    for x, y, w, h in faces:

        try:
            detected_emotion = DeepFace.analyze(frame, actions=['emotion'])

            img = cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

            cv2.rectangle(frame, (200, 400), (500, 500), (0, 0, 0), -1)
            cv2.putText(frame, detected_emotion[0]['dominant_emotion'], (230, 460), cv2.FONT_HERSHEY_COMPLEX, 2,
                        (0, 255, 0), 2, cv2.LINE_4)

            print(detected_emotion[0]['dominant_emotion'])

        except:
            print('no face')

    cv2.imshow('CAM', frame)
    key = cv2.waitKey(1)

    if key == ord('\x1b'):
        break

web_cam.release()
