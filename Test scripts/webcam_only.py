import cv2

##Video capture object
video_capture = cv2.VideoCapture(0)

while True:
    ret, frame = video_capture.read() #Getting data from web-cam

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) #Add another web-cam window, but in gray colour

    cv2.imshow('frame', frame)
    cv2.imshow('gray', gray)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video_capture.release()
cv2.destroyAllWindows()