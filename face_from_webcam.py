import cv2
import sys
import os

#cascPath = sys.argv[1]
faceCascade = cv2.CascadeClassifier('venv/lib/python3.6/site-packages/cv2/data/haarcascade_frontalface_default.xml')

face_id = input('\n enter user id end press <return> ==>  ') # For each person, enter one numeric face id
print("\n [INFO] Initializing face capture. Look the camera and wait ...")
# Initialize individual sampling face count
count = 0

video_capture = cv2.VideoCapture(0)
video_capture.set(3, 640) # set video width
video_capture.set(4, 480) # set video height
while True:
    # Capture frame-by-frame
    ret, frame = video_capture.read()

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30),
        flags=cv2.CASCADE_SCALE_IMAGE
    )

    # Draw a rectangle around the faces
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

        count += 1
        # Save the captured image into the dataset folder
        cv2.imwrite("dataset/User." + str(face_id) + '.' + str(count) + ".png", gray[y:y + h, x:x + w])

    # Display the resulting frame
    cv2.imshow('Video', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    elif count >= 100:  # Take 100 face samples and stop video
        break

# When everything is done, release the capture
video_capture.release()
cv2.destroyAllWindows()