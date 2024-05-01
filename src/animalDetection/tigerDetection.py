# Import libraries
import cv2
from pathlib import Path
# from ultralytics import YOLO
import yolov5



# Model path (assuming your model is named "best.pt" in the same folder)
model_path = Path("../../data/Animal/tiger.pt")  # Use Path object for clarity

# Load YOLOv5 model
model = yolov5.load(str(model_path))  # Convert Path object to string for YOLO

# Function to detect tigers in an image
def detect_tigers(frame):
    # Preprocess frame (optional, depending on your model's requirements)
    # You might need to resize or normalize the frame based on your model's input size

    # Perform YOLOv5 detection
    results = model(frame)
    # Get tiger detections (adjust class index based on your custom model classes)
    tiger_detections = results.pandas().xyxy[0][results.pandas().xyxy[0]['name'] == 'tiger']

    # Process detections (optional)
    # You can loop through detections, draw bounding boxes, etc.
    if not tiger_detections.empty:
        for index, row in tiger_detections.iterrows():
            xmin, ymin, xmax, ymax, confidence, class_name = row.to_list()
            cv2.rectangle(frame, (int(xmin), int(ymin)), (int(xmax), int(ymax)), (0, 0, 255), 2)  # Draw bounding box (blue)

    return frame

# Read video frame or image
cap = cv2.VideoCapture(0)  # Change to 0 for webcam or path to video file for video input
# cap = cv2.imread("image.jpg")  # For image input (replace with your image path)

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()

    # Detect tigers
    frame_with_detections = detect_tigers(frame)

    # Display the resulting frame
    cv2.imshow('Tiger Detection', frame_with_detections)

    # Exit if 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release capture
cap.release()
cv2.destroyAllWindows()
