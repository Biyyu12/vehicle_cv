import cv2
import supervision as sv
from ultralytics import YOLO
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
    track_activation_threshold=0.25,     
    lost_track_buffer=60,      
    minimum_matching_threshold=0.7
)

# Annotators
box_annotator = sv.BoxAnnotator()   
label_annotator = sv.LabelAnnotator()

# Line position
LINE_Y = 400

# counter
count_in = 0
count_out = 0

# memory for tracking history
track_history = {}

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

    results = model(frame, conf=0.4, iou=0.5)[0]
    detections = sv.Detections.from_ultralytics(results)
    
    mask = np.isin(detections.class_id, VEHICLE_CLASSES)
    detections = detections[mask]

    detections = tracker.update_with_detections(detections)

    for i in range(len(detections)):
        track_id = detections.tracker_id[i]
        x1, y1, x2, y2 = detections.xyxy[i]
        center_y = int((y1 + y2) / 2)

        if track_id not in track_history:
            track_history[track_id] = {
                "last_y": center_y,
                "counted": False
            }
            continue

        last_y = track_history[track_id]["last_y"]
        counted = track_history[track_id]["counted"]

        # Crossing logic
        if not counted:
            # masuk (atas → bawah)
            if last_y < LINE_Y <= center_y:
                count_in += 1
                track_history[track_id]["counted"] = True
            # keluar (bawah → atas)
            elif last_y > LINE_Y >= center_y:
                count_out += 1
                track_history[track_id]["counted"] = True

        track_history[track_id]["last_y"] = center_y

    cv2.line(frame, (0, LINE_Y), (frame.shape[1], LINE_Y), (0, 0, 255), 2) 

    labels = [
        f"ID {track_id}"
        for track_id in detections.tracker_id
    ]

    frame = box_annotator.annotate(frame, detections)
    frame = label_annotator.annotate(frame, detections, labels)

    # Display counts
    cv2.putText(frame, f"In: {count_in}", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.putText(frame, f"Out: {count_out}", (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Display FPS
    cv2.putText(frame, f"FPS: {int(fps)}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow("Vehicle Counting - Step 3", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()