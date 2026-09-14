import cv2
import numpy as np
import onnxruntime as ort


# COCO class names used by YOLOv8
CLASS_NAMES = [
    "person", "bicycle", "car", "motorcycle", "airplane",
    "bus", "train", "truck", "boat", "traffic light",
    "fire hydrant", "stop sign", "parking meter", "bench",
    "bird", "cat", "dog", "horse", "sheep",
    "cow", "elephant", "bear", "zebra", "giraffe",
    "backpack", "umbrella", "handbag", "tie", "suitcase",
    "frisbee", "skis", "snowboard", "sports ball", "kite",
    "baseball bat", "baseball glove", "skateboard", "surfboard",
    "tennis racket", "bottle", "wine glass", "cup", "fork",
    "knife", "spoon", "bowl", "banana", "apple",
    "sandwich", "orange", "broccoli", "carrot", "hot dog",
    "pizza", "donut", "cake", "chair", "couch",
    "potted plant", "bed", "dining table", "toilet", "tv",
    "laptop", "mouse", "remote", "keyboard", "cell phone",
    "microwave", "oven", "toaster", "sink", "refrigerator",
    "book", "clock", "vase", "scissors", "teddy bear",
    "hair drier", "toothbrush"
]


# Load ONNX model
session_options = ort.SessionOptions()
session_options.intra_op_num_threads = 1
session_options.inter_op_num_threads = 1
session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_BASIC

session = ort.InferenceSession(
    "yolov8n.onnx",
    sess_options=session_options,
    providers=["CPUExecutionProvider"]
)

input_name = session.get_inputs()[0].name


def letterbox(image, new_size=640):
    """
    Resize image while keeping the original aspect ratio.
    Adds padding if required.
    """

    height, width = image.shape[:2]

    scale = min(new_size / width, new_size / height)

    new_width = int(round(width * scale))
    new_height = int(round(height * scale))

    resized = cv2.resize(
        image,
        (new_width, new_height),
        interpolation=cv2.INTER_LINEAR
    )

    canvas = np.full(
        (new_size, new_size, 3),
        114,
        dtype=np.uint8
    )

    pad_x = (new_size - new_width) // 2
    pad_y = (new_size - new_height) // 2

    canvas[
        pad_y:pad_y + new_height,
        pad_x:pad_x + new_width
    ] = resized

    return canvas, scale, pad_x, pad_y


def detect_objects(image_bytes):

    # Convert uploaded bytes to OpenCV image
    np_array = np.frombuffer(image_bytes, np.uint8)
    frame = cv2.imdecode(np_array, cv2.IMREAD_COLOR)

    if frame is None:
        return None, {}

    original_height, original_width = frame.shape[:2]

    # Resize with letterbox
    input_image, scale, pad_x, pad_y = letterbox(frame, 640)

    # BGR -> RGB
    input_image = cv2.cvtColor(
        input_image,
        cv2.COLOR_BGR2RGB
    )

    # Normalize
    input_image = input_image.astype(np.float32) / 255.0

    # HWC -> CHW
    input_image = np.transpose(
        input_image,
        (2, 0, 1)
    )

    # Add batch dimension
    input_tensor = np.expand_dims(
        input_image,
        axis=0
    )

    # Run ONNX inference
    outputs = session.run(
        None,
        {input_name: input_tensor}
    )

    predictions = outputs[0][0]

    # YOLOv8 output:
    # 4 box values + 80 class scores
    predictions = predictions.T

    boxes = []
    scores = []
    class_ids = []

    confidence_threshold = 0.40
    nms_threshold = 0.45

    for prediction in predictions:

        x_center = prediction[0]
        y_center = prediction[1]
        width = prediction[2]
        height = prediction[3]

        class_scores = prediction[4:]

        class_id = int(np.argmax(class_scores))
        confidence = float(class_scores[class_id])

        if confidence < confidence_threshold:
            continue

        # Convert xywh -> x1,y1,x2,y2
        x1 = x_center - width / 2
        y1 = y_center - height / 2
        x2 = x_center + width / 2
        y2 = y_center + height / 2

        # Remove letterbox padding
        x1 = (x1 - pad_x) / scale
        y1 = (y1 - pad_y) / scale
        x2 = (x2 - pad_x) / scale
        y2 = (y2 - pad_y) / scale

        # Keep coordinates inside original image
        x1 = max(0, min(x1, original_width - 1))
        y1 = max(0, min(y1, original_height - 1))
        x2 = max(0, min(x2, original_width - 1))
        y2 = max(0, min(y2, original_height - 1))

        box_width = x2 - x1
        box_height = y2 - y1

        if box_width <= 0 or box_height <= 0:
            continue

        boxes.append([
            int(x1),
            int(y1),
            int(box_width),
            int(box_height)
        ])

        scores.append(confidence)
        class_ids.append(class_id)

    # Non-Maximum Suppression
    if boxes:

        indices = cv2.dnn.NMSBoxes(
            boxes,
            scores,
            confidence_threshold,
            nms_threshold
        )

        if len(indices) > 0:

            indices = np.array(indices).flatten()

            object_counts = {}

            for index in indices:

                x, y, w, h = boxes[index]

                class_id = class_ids[index]
                label = CLASS_NAMES[class_id]
                confidence = scores[index]

                # Count objects
                object_counts[label] = (
                    object_counts.get(label, 0) + 1
                )

                # Draw bounding box
                cv2.rectangle(
                    frame,
                    (x, y),
                    (x + w, y + h),
                    (0, 255, 0),
                    2
                )

                # Label
                text = f"{label} {confidence:.2f}"

                cv2.putText(
                    frame,
                    text,
                    (x, max(y - 10, 20)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 0),
                    2
                )

        else:
            object_counts = {}

    else:
        object_counts = {}

    # Convert annotated image to JPEG
    success, buffer = cv2.imencode(
        ".jpg",
        frame,
        [cv2.IMWRITE_JPEG_QUALITY, 80]
    )

    if not success:
        return None, {}

    return buffer.tobytes(), object_counts