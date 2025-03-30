#Imported cv library
import cv2

#Loading the pre trained models for facial detection
prototxt_path = 'deploy.prototxt' #Path to architecture file
model_path = 'res10_300x300_ssd_iter_140000_fp16.caffemodel' #Path to model file

#Loading the model
net = cv2.dnn.readNetFromCaffe(prototxt_path, model_path)

#Setting the backend and target
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

#Initializing the webcam
cap = cv2.VideoCapture(0)

#Loading the Haar Cascade classifier for detection
face_casscade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

#Capturing frame by frame
while True:
    
    ret, frame = cap.read()
    
    if not ret:
        break

    h, w = frame.shape[:2]

    #Converting image to blob
    blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300), (104.0, 177.0, 123.0))

    net.setInput(blob)
    detections = net.forward()

    #Looping through detected faces
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence>0.5:
            box = detections[0, 0, i, 3:7] * [w, h, w, h]
            (x, y, x_max, y_max) = box.astype("int")
            cv2.rectangle(frame, (x, y), (x_max, y_max), (255, 0, 0), 2)
            cv2.putText(frame, f"{confidence*100:.2f}%", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

    #Displaying the output
    cv2.imshow('Face Detection', frame)

    #Conditions for the closing of program
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

#Releasing the webcam and destroying all windows
cap.release()
cv2.destroyAllWindows()

#End of code