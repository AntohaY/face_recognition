import cv2
import numpy as np
import os

recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read('trainer/trainer.yml')

faceCascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml');
font = cv2.FONT_HERSHEY_SIMPLEX
# iniciate id counter
id = 0

# names related to ids: example ==> Marcelo: id=1,  etc
names = ['None', 'Anton', 'Lox']

# Initialize and start realtime video capture
video_capture = cv2.VideoCapture(0)
video_capture.set(3, 640)  # set video widht
video_capture.set(4, 480)  # set video height

# Define min window size to be recognized as a face
minW = 0.1 * video_capture.get(3)
minH = 0.1 * video_capture.get(4)

while True:
    ret, frame = video_capture.read()

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(int(minW), int(minH)),
    )
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        id, confidence = recognizer.predict(gray[y:y + h, x:x + w])
        #Confidence(100 is better than 200, and 0 would be a "perfect match"))
        if (confidence < 100):
            id = names[id]
            confidence = "  {0}%".format(round(100 - confidence))
        else:
            id = "unknown"
            confidence = "  {0}%".format(round(100 - confidence))

        cv2.putText(frame, str(id), (x + 5, y - 5), font, 1, (255, 255, 255), 2)
        cv2.putText(frame, str(confidence), (x + 5, y + h - 5), font, 1, (255, 255, 0), 1)

    cv2.imshow('Video', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):  # Press 'q' for exit
        break
# Do a bit of cleanup
print("\n [INFO] Exiting Program and cleanup stuff")
video_capture.release()
cv2.destroyAllWindows()