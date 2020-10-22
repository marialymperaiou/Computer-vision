import cv2

# Loading the cascades
# for the whole face
face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml") #filters to detect the face
# for the eyes
eye_cascade = cv2.CascadeClassifier("haarcascade_eye.xml") #filters to detect the face
# for smile
smile_cascade = cv2.CascadeClassifier("haarcascade_smile.xml")

# Function for detections
def detect(gray, frame):    
    # image, scale factor, minimum number of zones
    faces = face_cascade.detectMultiScale(gray, 1.3, 5) # coordinates
    
    # (x, y, w, h) of the rectangle of the frame for the face
    # they correspond to the upper left (x, y) coordinates and
    # the width and the height of the rectangle respectively
    for (x, y, w, h) in faces:
        # arguments are: frame, upper left corner of the rectangle, lower right corner
        # line color, line width
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
        # for gray image
        roi_gray = gray[y: y + h, x: x + w]
        # for RGB image
        roi_color = frame[y: y + h, x: x + w]
        # detect eyes
        eyes = eye_cascade.detectMultiScale(roi_gray, 1.1, 3)
        # detect smile
        smile = smile_cascade.detectMultiScale(roi_gray, 1.4, 10)
                                            
        
        # draw the rectangles around the eyes
        for (ex, ey, ew, eh) in eyes:
            cv2.rectangle(roi_color, (ex, ey), (ex + ew,ey + eh), (0, 255, 0), 2)
            
        # draw the rectangles around the smile
        for (sx, sy, sw, sh) in smile:
            cv2.rectangle(roi_color, (sx, sy), (sx + sw,sy + sh), (0, 0, 255), 2)
        
    return frame

# Face recognition from webcam
video_capture = cv2.VideoCapture(0) # 0 from internal computer webcam, 1 from external device
while True:
    # frame is the original color image
    _, frame = video_capture.read()
    # B&W version of the original image (frame)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) # averages blue, green and red
    canvas = detect(gray, frame)
    # display all the successive outputs (processed images with rectangles)
    cv2.imshow('Video', canvas)
    # Stop the webcam and the face detection process
    if cv2.waitKey(1) & 0xFF == ord('q'): # if 'q' is pressed on keyboard
        break
    
# Turn off the webcan
video_capture.release()
cv2.destroyAllWindows()
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        