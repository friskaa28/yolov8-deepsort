from ultralytics import YOLO
import cv2
import numpy as np
import math
from sort import Sort  # Import the Sort class from the sort module

# Initialize webcam
cap = cv2.VideoCapture(0)

# Get frame dimensions
frame_width = int(cap.get(3))
frame_height = int(cap.get(4))

# Initialize video writer
out = cv2.VideoWriter('output.avi', cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), 60, (frame_width, frame_height))

#30 adalah frame rate, untuk pengujian bisa dilakukan untuk frame rate 15,30 dan 60 fps

# Load YOLO model
model = YOLO('../YOLO-Weights/detection_model_optimization.pt')

# Define class names
classNames = ["Brankar", "CT-SCAN", "Defibrilator", "Dialysis Machine", "Glukometer", "Hand Sanitizer", "Hospital Bed",
              "Insulin Pen", "Kursi Roda", "Mask", "Stetoskop", "Tensimeter", "Termometer", "Tiang Infus",
              "Timbangan Badan", "X-Ray"]

# Initialize SORT tracker with adjusted parameters to reduce ID switches
tracker = Sort(max_age=10, min_hits=3, iou_threshold=0.3)  # Adjusted parameters

# Define detection threshold bisa dibuat start 0.3-0.5 (low) dan 0.5-0.7 (high)
CONF_THRESHOLD = 0.6

while cap.isOpened():
    success, img = cap.read()
    if success:
        # Perform object detection
        results = model(img, stream=True)

        # Update detections
        detections = []
        for r in results:
            boxes = r.boxes  # Get bounding boxes
            for box in boxes:
                conf = box.conf[0]
                if conf >= CONF_THRESHOLD:  # Only consider detections above threshold
                    x1, y1, x2, y2 = box.xyxy[0]
                    cls = int(box.cls[0])
                    if cls < len(classNames):  # Filter only the desired classes
                        detection = [x1, y1, x2, y2, conf, cls]  # Create a list for each detection
                        detections.append(detection)

        # Convert detections to numpy array
        np_detections = np.array(detections)

        # Update tracker with current frame detections
        tracks = tracker.update(np_detections)

        # Draw tracked objects
        for track in tracks:
            x1, y1, x2, y2, track_id = track.astype(int)
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(img, f'Tracked ID: {track_id}', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            # Retrieve the detection for this track_id
            detection_idx = next((i for i, d in enumerate(detections) if np.array_equal(d[:4], track[:4])), None)
            if detection_idx is not None:
                conf = math.ceil((detections[detection_idx][4] * 100)) / 100
                cls = int(detections[detection_idx][5])
                class_name = classNames[cls]
                label = f'{class_name} {conf}'
                t_size = cv2.getTextSize(label, 0, fontScale=1, thickness=2)[0]
                c2 = x1 + t_size[0], y1 - t_size[1] - 3
                cv2.rectangle(img, (x1, y1), c2, [255, 0, 255], -1, cv2.LINE_AA)
                cv2.putText(img, label, (x1, y1 - 2), 0, 1, [255, 255, 255], thickness=1, lineType=cv2.LINE_AA)

        # Write frame to output video
        out.write(img)
        cv2.imshow("Image", img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
out.release()
cv2.destroyAllWindows()
