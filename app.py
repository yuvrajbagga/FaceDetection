import cv2

#first of all we need to measure face boints like eyes nose etc with the help of cascadeclassifierfunc
face_capture=cv2.CascadeClassifier("frontface.xml")

#to enable the video camera on realtime
video_camera=cv2.VideoCapture(0)
while True :
#now create 3 variable to read the image
    ret,video_data=video_camera.read()
#now we need to black and white our photo to detect muscles and revert to color image
    col=cv2.cvtColor(video_data,cv2.COLOR_BGR2GRAY)
#now we apply function to measure face structure
    faces=face_capture.detectMultiScale(
        col,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30),
        flags=cv2.CASCADE_SCALE_IMAGE
    )
    
#now create face frame or box
    for (x,y,w,h) in faces:
        cv2.rectangle(video_data,(x,y),(x+w,y+h),(0,255,0),2)

#now create screen 
    cv2.imshow("Face Detection",video_data)
#now we need to stop the video
    if cv2.waitKey(10) == ord(" "):
        break
#now we run the camera
video_camera.release() 



