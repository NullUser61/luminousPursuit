import cv2
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# Replace 'yolov8n.pt' with the path to your pre-trained YOLOv8 model weights file
model = torch.hub.load('../../data/Animal/Tiger', 'yolov8n', pretrained=True)

# Set the class index for tiger (assuming 'tiger' is class 0 in your dataset)
class_index = 0

# Open the default webcam (replace 0 with a different video source index if needed)
cap = cv2.VideoCapture(0)

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()

    # Run inference on the frame
    results = model(frame)

    # Get detections
    detections = results.pandas().xyxy[0]

    # Check if any tigers are detected
    tigers_detected = detections[detections['name'] == 'tiger']

    # Draw bounding boxes and labels for detected tigers
    if not tigers_detected.empty:
        for index, row in tigers_detected.iterrows():
            xmin, ymin, xmax, ymax, confidence, name = row.values
            cv2.rectangle(frame, (int(xmin), int(ymin)), (int(xmax), int(ymax)), (0, 0, 255), 2)
            cv2.putText(frame, f"{name} ({confidence:.2f})", (int(xmin), int(ymin) - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

    # Display the resulting frame
    cv2.imshow('Tiger Detection', frame)

    # Exit on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release capture and close all windows
cap.release()
cv2.destroyAllWindows()
