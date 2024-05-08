import cv2
from ultralytics import YOLO
def animal(frame):

    model = YOLO("data/Animal/besttiger.pt")  # load a custom model
    

    threshold = 0.7
    results = model(frame)[0]

    for result in results.boxes.data.tolist():
        x1, y1, x2, y2, score, class_id = result

        if score > threshold:
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 4)
            cv2.putText(frame, results.names[int(class_id)].upper(), (int(x1), int(y1 - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 255, 0), 3, cv2.LINE_AA)
    return frame
# Display the captured frame
    

if __name__ == "__main__":
    animal()
