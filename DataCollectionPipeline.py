import cv2
import os
import mediapipe as mp

# Create directories for storing images
def create_directories():
    os.makedirs("data/test2/closed", exist_ok=True)
    os.makedirs("data/test2/opened", exist_ok=True)
    
# Save cropped eye image
def save_image(image, label, side, count):
    """
    Saves the image into the specified label folder with a filename including side info.

    Parameters:
    - image: Image to save.
    - label: "opened" or "closed" to specify the state of the eyes.
    - side: "left" or "right" to specify the eye side.
    - count: The unique count for the image filename.
    """
    directory = f"data/test2/{label}"
    filename = f"{directory}/{side}_eye_{count}.jpg"
    os.makedirs(directory, exist_ok=True)
    cv2.imwrite(filename, image)
    print(f"Saved: {filename}")


# Extract eye regions from landmarks
def crop_eye(frame, landmarks, eye_indices, margin=20):
    """
    Crops the eye region with additional margin and resizes it.

    Parameters:
    - frame: The input image frame.
    - landmarks: Facial landmarks detected by Mediapipe.
    - eye_indices: Indices of the landmarks for the eye region.
    - margin: Extra pixels to add around the eye bounding box.

    Returns:
    - Cropped and resized image of the eye with margin.
    """
    h, w, _ = frame.shape
    x_min = int(min([landmarks[i].x * w for i in eye_indices]) - margin)
    x_max = int(max([landmarks[i].x * w for i in eye_indices]) + margin)
    y_min = int(min([landmarks[i].y * h for i in eye_indices]) - margin)
    y_max = int(max([landmarks[i].y * h for i in eye_indices]) + margin)

    # Ensure the bounding box stays within the frame
    x_min = max(x_min, 0)
    y_min = max(y_min, 0)
    x_max = min(x_max, w)
    y_max = min(y_max, h)

    # Crop the eye region
    cropped_eye = frame[y_min:y_max, x_min:x_max]

    # Resize to 224x224
    resized_eye = cv2.resize(cropped_eye, (224, 224))
    return resized_eye


# Main function for capturing eye images
def collect_eye_images():
    create_directories()

    # Initialize Mediapipe Face Mesh
    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1)

    # Start video capture
    cap = cv2.VideoCapture(1)
    if not cap.isOpened():
        print("Error: Could not open the camera.")
        return

    print("Press 'o' for open eyes, 'c' for closed eyes, and 'q' to quit.")

    open_count, closed_count = 1000, 1000

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Unable to capture frame.")
            break

        # Flip frame horizontally for a mirrored view
        frame = cv2.flip(frame, 1)

        # Convert frame to RGB for Mediapipe processing
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb_frame)

        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                # Indices for the left and right eyes
                left_eye_indices = [33, 160, 158, 133, 153, 144]
                right_eye_indices = [362, 385, 387, 263, 373, 380]

                # Crop the left and right eye regions
                left_eye = crop_eye(frame, face_landmarks.landmark, left_eye_indices, margin=30)
                right_eye = crop_eye(frame, face_landmarks.landmark, right_eye_indices, margin=30)

                if left_eye.size > 0:
                    cv2.imshow("Left Eye", left_eye)
                if right_eye.size > 0:
                    cv2.imshow("Right Eye", right_eye)

                # Display instructions
                cv2.putText(frame, "Press 'o' for Open Eyes, 'c' for Closed Eyes, 'q' to Quit",
                            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
                cv2.imshow("Eye Dataset Collection", frame)

                # Get user input
                key = cv2.waitKey(1) & 0xFF

                if key == ord('o'):  # Save open eye images
                    if left_eye.size > 0:
                        save_image(left_eye, "opened", "left", open_count)
                    if right_eye.size > 0:
                        save_image(right_eye, "opened", "right", open_count)
                    open_count += 1

                elif key == ord('c'):  # Save closed eye images
                    if left_eye.size > 0:
                        save_image(left_eye, "closed", "left", closed_count)
                    if right_eye.size > 0:
                        save_image(right_eye, "closed", "right", closed_count)
                    closed_count += 1

                elif key == ord('q'):  # Quit the program
                    cap.release()
                    cv2.destroyAllWindows()
                    return

    # Release resources
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    collect_eye_images()
