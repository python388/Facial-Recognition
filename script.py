import cv2
import os
from ultralytics import YOLO

# --- Configuration ---
MODEL_NAME = "yolov8n.pt"  # Nano model - fast and small
ENGINE_NAME = "yolov8n.engine"
PERSON_CLASS_ID = 0

def get_optimized_model():
    """
    Checks if a TensorRT engine exists. If not, it builds one.
    Returns the optimized YOLO model object.
    """
    if os.path.exists(ENGINE_NAME):
        print(f"Loading existing TensorRT engine: {ENGINE_NAME}")
        model = YOLO(ENGINE_NAME, task='detect')
    else:
        print(f"No TensorRT engine found. Building one from {MODEL_NAME}...")
        print("This may take 5-15 minutes on a Jetson Nano. Please be patient.")
        
        # Load the base PyTorch model
        model = YOLO(MODEL_NAME)
        
        # Export the model to TensorRT format
        # This is the key optimization step!
        # It ensures the model runs on the GPU with maximum efficiency.
        # 'data=None' and 'half=True' are good defaults for Jetson
        model.export(format="engine", data=None, half=True, imgsz=640)
        
        # The exported model is saved as MODEL_NAME.engine (e.g., "yolov8n.engine")
        # We rename it just to be clean
        if os.path.exists(MODEL_NAME.replace('.pt', '.engine')):
            os.rename(MODEL_NAME.replace('.pt', '.engine'), ENGINE_NAME)
        
        print(f"Engine built and saved as {ENGINE_NAME}")
        # Load the newly created engine
        model = YOLO(ENGINE_NAME, task='detect')
        
    return model

def gstreamer_pipeline(
    capture_width=1280,
    capture_height=720,
    display_width=1280,
    display_height=720,
    framerate=30,
    flip_method=0,
):
    """
    Creates a GStreamer pipeline string for capturing from a CSI camera
    on a Jetson Nano. Use this for the Pi Camera.
    
    For a USB webcam, just use `cv2.VideoCapture(0)` or `cv2.VideoCapture(1)`.
    """
    return (
        "nvarguscamerasrc ! "
        "video/x-raw(memory:NVMM), "
        "width=(int)%d, height=(int)%d, "
        "format=(string)NV12, framerate=(fraction)%d/1 ! "
        "nvvidconv flip-method=%d ! "
        "video/x-raw, width=(int)%d, height=(int)%d, format=(string)BGRx ! "
        "videoconvert ! "
        "video/x-raw, format=(string)BGR ! appsink"
        % (
            capture_width,
            capture_height,
            framerate,
            flip_method,
            display_width,
            display_height,
        )
    )

def run_detection():
    """
    Main loop to run person detection on a webcam feed.
    """
    try:
        model = get_optimized_model()
    except Exception as e:
        print(f"Error loading/building model: {e}")
        print("Please ensure you have an internet connection to download the model.")
        return

    # --- CHOOSE YOUR CAMERA ---
    # 1. For a USB Webcam:
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    # 2. For the Raspberry Pi CSI Camera:
    # Uncomment the line below to use the GStreamer pipeline
    # pipeline = gstreamer_pipeline(flip_method=0)
    # cap = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)
    # --------------------------

    if not cap.isOpened():
        print("Error: Could not open camera.")
        return

    print("\nStarting detection loop. Press 'q' to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to grab frame.")
            break

        # Run inference on the frame
        # 'verbose=False' stops it from printing results to console
        results = model(frame, verbose=False, device=0)

        # Process the results
        for result in results:
            boxes = result.boxes
            for box in boxes:
                # Check if the detected object is a person
                if int(box.cls) == PERSON_CLASS_ID:
                    # Get bounding box coordinates
                    x1, y1, x2, y2 = [int(i) for i in box.xyxy[0]]
                    
                    # Get confidence score
                    conf = float(box.conf)
                    
                    # Draw rectangle and label
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    label = f"Person: {conf:.2f}"
                    cv2.putText(
                        frame,
                        label,
                        (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (0, 255, 0),
                        2,
                    )

        # Display the resulting frame
        cv2.imshow("Optimized Person Detection (Jetson Nano)", frame)

        # Check for 'q' key to exit
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    # Release everything
    cap.release()
    cv2.destroyAllWindows()
    print("Detection stopped.")

if __name__ == "__main__":
    run_detection()
