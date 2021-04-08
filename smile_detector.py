import cv2

#Face and smile classifiers
face_detector = cv2.CascadeClassifier(cv2.data.haarcascades+'haarcascade_frontalface_default.xml')
smile_detector = cv2.CascadeClassifier(cv2.data.haarcascades+'haarcascade_smile.xml')

#Grab Webcam feed
webcam = cv2.VideoCapture(0)

#Show the current frame
while True:
  #Read the current frame from the webcam video stream
  successful_frame_read, frame = webcam.read()

  #If there's an error, abort
  if not successful_frame_read:
    break

  #Change to Grayscale
  frame_grayscale = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
  #Detect faces 
  faces = face_detector.detectMultiScale(frame_grayscale)

  #Run face detection
  for (x, y, w, h) in faces:
    #Draw a rectangle around the face
    cv2.rectangle(frame, (x, y), (x+w, y+h), (100, 200, 50), 4)

    #Get the sub frame (using numpy N-dimensional array slicing)
    the_face = frame[y:y+h, x:x+w]
    #Change to Grayscale
    face_grayscale = cv2.cvtColor(the_face, cv2.COLOR_BGR2GRAY)

    #Detect smiles
    smiles = smile_detector.detectMultiScale(face_grayscale, scaleFactor=1.7, minNeighbors=20)
    # #Find the smile in the face
    # for(x_, y_, w_, h_) in smiles:

    #   #Draw a rectangle around the smile
    #   cv2.rectangle(the_face, (x_, y_), (x_ + w_, y_ + h_), (50, 50, 200), 4)
    
    #Label the face as smiling
    if len(smiles) > 0:
      cv2.putText(frame, 'Smiling', (x, y+h+40), fontScale=0.8, fontFace=cv2.FONT_HERSHEY_DUPLEX, color=(255, 255, 255))

  #Show the current frame
  cv2.imshow('Smile Detector', frame)
  #Display
  cv2.waitKey(10)

webcam.release()
cv2.destroyAllWindows() 
