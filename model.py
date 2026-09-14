from ultralytics import YOLO
import cv2
import numpy as np

# Load YOLOv8 model
model = YOLO("yolov8n.pt")


def detect_objects(image_bytes):
    # Convert uploaded image bytes into an OpenCV image
    np_array = np.frombuffer(image_bytes, np.uint8)
    frame = cv2.imdecode(np_array, cv2.IMREAD_COLOR)

    if frame is None:
        return None, {}

    # Run YOLO detection
    results = model(frame)

    # Get detected objects
    boxes = results[0].boxes

    # Store object counts
    object_counts = {}

    for box in boxes:
        cls_id = int(box.cls[0])
        label = model.names[cls_id]

        object_counts[label] = object_counts.get(label, 0) + 1

    # Draw bounding boxes and labels
    annotated_frame = results[0].plot()

    # Convert image to JPEG
    success, buffer = cv2.imencode(".jpg", annotated_frame)

    if not success:
        return None, {}

    return buffer.tobytes(), object_counts