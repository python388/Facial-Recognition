import cv2
import numpy as np
from deepface import DeepFace

# --- Configuration (Assuming CAMERA_INDEX = 0) ---
CAMERA_INDEX = 0  
MODEL_NAME = "mtcnn" 
TARGET_SIZE = (224, 224) 

# **NEW: State Flag Variable**
face_is_present = False 

print("DeepFace Camera Detector Initializing...")
cap = cv2.VideoCapture(CAMERA_INDEX)

if not cap.isOpened():
    print(f"Error: Could not open camera with index {CAMERA_INDEX}.")
    exit()

print("Camera opened successfully. Press 'q' to exit.")

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to grab frame.")
            break

        face_detected_in_current_frame = False

        try:
            # Attempt face detection
            face_objects = DeepFace.extract_faces(
                img_path=frame, 
                target_size=TARGET_SIZE, 
                detector_backend=MODEL_NAME, 
                enforce_detection=True
            )

            if len(face_objects) > 0:
                face_detected_in_current_frame = True
                
                # Draw box (optional visual feedback)
                x, y, w, h = face_objects[0]['facial_area']['x'], \
                             face_objects[0]['facial_area']['y'], \
                             face_objects[0]['facial_area']['w'], \
                             face_objects[0]['facial_area']['h']
                             
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            
        except ValueError:
            # No face detected in the current frame, so the flag remains False
            pass
        except Exception as e:
            print(f"An error occurred during face processing: {e}")
            
        
        # --- LOGIC TO PRINT ONLY ONCE ---
        
        # Scenario 1: Face just appeared (Transition from False to True)
        if face_detected_in_current_frame and not face_is_present:
            print("\n>> face detected <<")
            face_is_present = True # Update state to prevent immediate re-printing
            
        # Scenario 2: Face just left (Optional: Print when face leaves)
        elif not face_detected_in_current_frame and face_is_present:
            print(">> face lost <<")
            face_is_present = False # Update state to allow printing again later

        # Display the resulting frame (optional)
        cv2.imshow('DeepFace Detector', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    cap.release()
    cv2.destroyAllWindows()
    print("Application closed.")