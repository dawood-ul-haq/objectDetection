from ultralytics import YOLO
import cv2

model = YOLO("yolov8n.pt")

camera = None
is_running = False

# Store object counts
object_counts = {}

def start_camera():
    global camera, is_running
    if camera is None:
        camera = cv2.VideoCapture(0)
    is_running = True

def stop_camera():
    global camera, is_running
    is_running = False
    if camera:
        camera.release()
        camera = None

def get_counts():
    return object_counts

def generate_frames():
    global camera, is_running, object_counts

    while True:
        if not is_running or camera is None:
            continue

        success, frame = camera.read()
        if not success:
            break

        results = model(frame)
        boxes = results[0].boxes

        object_counts = {}  # reset per frame

        for box in boxes:
            cls_id = int(box.cls[0])
            label = model.names[cls_id]

            object_counts[label] = object_counts.get(label, 0) + 1

        annotated_frame = results[0].plot()

        ret, buffer = cv2.imencode('.jpg', annotated_frame)
        frame_bytes = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')