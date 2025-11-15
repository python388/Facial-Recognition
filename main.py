import cv2
import numpy as np
from deepface import DeepFace

# --- Configuration ---
CAMERA_INDEX = 0  # Change this to 1 or a GStreamer pipeline string for CSI camera
MODEL_NAME = "mtcnn" # 'mtcnn' is often a good balance for speed/accuracy for detection. 
                     # Other options: 'opencv', 'ssd', 'dlib', 'retinaface'
TARGET_SIZE = (224, 224) # DeepFace models expect a certain input size

print("DeepFace Camera Detector Initializing...")

# Initialize the camera
# For Jetson Nano CSI camera, you might use a gstreamer pipeline here instead of 0
cap = cv2.VideoCapture(CAMERA_INDEX)

if not cap.isOpened():
    print(f"Error: Could not open camera with index {CAMERA_INDEX}.")
    exit()

print("Camera opened successfully. Press 'q' to exit.")

try:
    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()
        
        if not ret:
            print("Error: Failed to grab frame.")
            break

        # Check for face detection using DeepFace
        # We wrap it in a try-except block because DeepFace throws a ValueError
        # when no face is detected, which is the normal behavior we are checking for.
        try:
            # We use enforce_detection=False to avoid a crash when no face is found,
            # but DeepFace still raises a ValueError if it can't find a face
            # that meets its quality/size criteria.
            # Using 'detector_backend' to specify the face detection model.
            
            # The 'extract_faces' function is one of the lightest DeepFace calls for 
            # *only* face detection and alignment. It returns a list of faces found.
            
            face_objects = DeepFace.extract_faces(
                img_path=frame, 
                target_size=TARGET_SIZE, 
                detector_backend=MODEL_NAME, 
                enforce_detection=True # Set to False to try and avoid error, but 
                                      # DeepFace's behavior with just detection can vary. 
                                      # We use the ValueError catch for simplicity.
            )

            if len(face_objects) > 0:
                print("face detected")
                
                # Optional: Draw a box around the first detected face for visual feedback
                x, y, w, h = face_objects[0]['facial_area']['x'], \
                             face_objects[0]['facial_area']['y'], \
                             face_objects[0]['facial_area']['w'], \
                             face_objects[0]['facial_area']['h']
                             
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            
        except ValueError as e:
            # This is the common DeepFace error when no face is detected in the frame
            # print("No face detected in the current frame.")
            pass
        except Exception as e:
            # Catch other potential errors during processing
            print(f"An error occurred during face processing: {e}")
            
        # Display the resulting frame (optional, comment out if running headless)
        cv2.imshow('DeepFace Detector', frame)

        # Break the loop on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    # When everything is done, release the capture and destroy windows
    cap.release()
    cv2.destroyAllWindows()
    print("Application closed.")