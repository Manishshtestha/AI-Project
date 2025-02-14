import cv2
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

# Load the pre-trained model
model = load_model('ML_MODEL/Men_Women_model_from_scratch.h5')

# Initialize the face detector (Haar Cascade for face detection)
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Start video capture from the webcam
cap = cv2.VideoCapture(0)

# Define the font and color for displaying text on the image
font = cv2.FONT_HERSHEY_SIMPLEX
color = (255, 0, 0)  # Blue text
font_scale = 1
thickness = 2

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    
    # Convert the frame to grayscale (Haar cascade needs grayscale images)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the frame
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    # Loop over each face detected
    for (x, y, w, h) in faces:
        # Draw a rectangle around the face (bounding box)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)  # Green rectangle

        # Crop the face from the frame and resize it to 128x128 for the model
        face_img = frame[y:y+h, x:x+w]
        face_resized = cv2.resize(face_img, (128, 128))

        # Convert the face image to the format that Keras expects
        img_array = image.img_to_array(face_resized) / 255.0  # Normalize the image
        img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension

        # Make a prediction
        prediction = model.predict(img_array)

        # Determine whether the prediction is a dog or a cat
        if prediction >= 0.5:
            label = f"Male {prediction}"
            label_color = (0, 0, 255)  # Red color for Dog
        else:
            label = f"Female {prediction}"
            label_color = (255, 0, 0)  # Blue color for Cat

        # Calculate the position to put the label inside the bounding box
        label_position = (x + 10, y + 30)

        # Put the label inside the bounding box
        cv2.putText(frame, label, label_position, font, font_scale, label_color, thickness)

    # Display the image with bounding boxes and labels
    cv2.imshow('Gender Detection with Face Bounding Box', frame)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture object and close all windows
cap.release()
cv2.destroyAllWindows()
