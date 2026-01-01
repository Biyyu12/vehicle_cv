import cv2
from ultralytics import YOLO
import supervision as sv
import numpy as np

# Load YOLOv8
model = YOLO("yolov8n.pt")

# Video input
cap = cv2.VideoCapture("assets/traffic.mp4")

# Menghitung fps
tick_freq = cv2.getTickFrequency()
prev_tick = cv2.getTickCount()

# ByteTrack tracker
tracker = sv.ByteTrack(
    track_activation_threshold=0.3,     
    lost_track_buffer=60,      
    minimum_matching_threshold=0.7
)

# Annotators
box_annotator = sv.BoxAnnotator()
label_annotator = sv.LabelAnnotator()

# Vehicle classes
VEHICLE_CLASSES = [2, 3, 5, 7]

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    # Calculate FPS
    current_tick = cv2.getTickCount()
    time_diff = (current_tick - prev_tick) / tick_freq
    fps = 1 / time_diff if time_diff > 0 else 0
    prev_tick = current_tick

    # YOLO detection
    results = model(frame, conf=0.4, iou=0.5)[0]

    # Convert to Supervision Detections
    detections = sv.Detections.from_ultralytics(results)

    # Filter vehicle classes
    mask = np.isin(detections.class_id, VEHICLE_CLASSES)
    detections = detections[mask]

    # Tracking
    detections = tracker.update_with_detections(detections)

    # Labels with ID
    labels = [
        f"ID {track_id} | {model.names[class_id]}"
        for track_id, class_id in zip(
            detections.tracker_id,
            detections.class_id
        )
    ]

    # Draw
    frame = box_annotator.annotate(frame, detections)
    frame = label_annotator.annotate(frame, detections, labels)

    # Display FPS
    cv2.putText(frame, f"FPS: {int(fps)}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.imshow("Vehicle Tracking - Step 2", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
