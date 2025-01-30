import cv2
import numpy as np
import mediapipe as mp
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Load the trained model
model = tf.keras.models.load_model('test_model.h5')

# Initialize MediaPipe face mesh and drawing utilities
mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils

def crop_eye(image, landmarks, eye_points, margin=15):
    eye = np.array([(landmarks.landmark[i].x, landmarks.landmark[i].y) for i in eye_points])
    height, width, _ = image.shape
    eye = eye * [width, height]
    x_min = int(np.min(eye[:, 0])) - margin
    y_min = int(np.min(eye[:, 1])) - margin
    x_max = int(np.max(eye[:, 0])) + margin
    y_max = int(np.max(eye[:, 1])) + margin
    return image[max(0, y_min):y_max, max(0, x_min):x_max]

def preprocess_eye_image(cropped_eye):
    # Resize the image to (224, 224)
    resized_eye = cv2.resize(cropped_eye, (224, 224))

    # Rescale the pixel values to [0, 1]
    normalized_eye = resized_eye / 255.0

    # Add batch dimension (1, 224, 224, 3)
    return np.expand_dims(normalized_eye, axis=0)

def predict_eye_state(model, eye_image):
    eye_input = preprocess_eye_image(eye_image)
    prediction = model.predict(eye_input)
    print(prediction)
    predicted_class = np.argmax(prediction, axis=1)[0]
    return "open" if predicted_class == 1 else "closed"

# Initialize webcam
cap = cv2.VideoCapture(1)

with mp_face_mesh.FaceMesh(min_detection_confidence=0.5, min_tracking_confidence=0.5) as face_mesh:
    last_state, total_blinks = "open", 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Convert the frame to RGB
        # image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Process the frame with MediaPipe FaceMesh
        results = face_mesh.process(frame)

        if results.multi_face_landmarks:
            for landmarks in results.multi_face_landmarks:
                left_eye_points = [33, 133, 160, 158, 159, 145]
                right_eye_points = [362, 263, 387, 385, 386, 374]

                left_eye_crop = crop_eye(frame, landmarks, left_eye_points, margin=15)
                right_eye_crop = crop_eye(frame, landmarks, right_eye_points, margin=15)

                if left_eye_crop.size == 0 or right_eye_crop.size == 0:
                    continue

                # Predict eye states
                current_state_left = predict_eye_state(model, left_eye_crop)
                current_state_right = predict_eye_state(model, right_eye_crop)
                current_state = "closed" if current_state_left == "closed" and current_state_right == "closed" else "open"

                if last_state == "open" and current_state == "closed":
                    total_blinks += 1
                    print(f"Blink detected! Total blinks: {total_blinks}")

                last_state = current_state

                # Display prediction on the screen
                text = f"Eye State: {current_state} | Total Blinks: {total_blinks}"
                cv2.putText(frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # Draw face mesh landmarks
        if results.multi_face_landmarks:
            for landmarks in results.multi_face_landmarks:
                mp_drawing.draw_landmarks(
                    frame, landmarks, mp_face_mesh.FACEMESH_CONTOURS
                )

        # Display the frame
        cv2.imshow('Live Face Mesh', frame)

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

# Release resources
cap.release()
cv2.destroyAllWindows()
