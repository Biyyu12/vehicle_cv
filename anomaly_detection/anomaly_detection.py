import cv2
import supervision as sv
from ultralytics import YOLO
import numpy as np
import time 
from collections import deque

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
    lost_track_buffer=120,      
    minimum_matching_threshold=0.7
)

# Annotators
box_annotator = sv.BoxAnnotator()   
label_annotator = sv.LabelAnnotator()

# Line position
LINE_Y = 400

# constants
PIXEL_TO_METER = 0.05
MIN_PIXEL_MOVE = 15
WINDOW_SIZE = 15
ZONE_OFFSET = 30
ZONE_TOP = LINE_Y - ZONE_OFFSET
ZONE_BOTTOM = LINE_Y + ZONE_OFFSET
MAX_TRACK_AGE = 300

# constants anomaly detection
MAX_SPEED = 80
MIN_SPEED = 3
STOP_TIME_THRESHOLD = 3.0

# counter
count_in = 0
count_out = 0

# memory for tracking history
track_history = {}

# Vehicle classes
VEHICLE_CLASSES = [2, 3, 5, 7]

# fungsi smoothing posisi
def smooth_position(history):
    ys = [p[0] for p in history]
    return int(np.mean(ys))

frame_idx = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    frame_idx += 1

    # Calculate FPS
    current_tick = cv2.getTickCount()
    time_diff = (current_tick - prev_tick) / tick_freq
    fps = 1 / time_diff if time_diff > 0 else 0
    prev_tick = current_tick

    results = model(frame, conf=0.3, iou=0.5)[0]
    detections = sv.Detections.from_ultralytics(results)
    
    mask = np.isin(detections.class_id, VEHICLE_CLASSES)
    detections = detections[mask]

    detections = tracker.update_with_detections(detections)

    for i in range(len(detections)):
        track_id = detections.tracker_id[i]
        x1, y1, x2, y2 = detections.xyxy[i]
        center_y = int((y1 + y2) / 2)
        current_time = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0

        if track_id not in track_history:
            track_history[track_id] = {
                "last_y": center_y,
                "counted": False,
                "positions": deque(maxlen=WINDOW_SIZE),
                "speed": 0.0,
                "zone_history": deque(maxlen=5),
                "last_seen": frame_idx,
                "anomaly": None,
                "last_move_time": current_time,
                "direction": None,
            }
            track_history[track_id]["positions"].append((center_y, current_time))
            continue

        track_history[track_id]["positions"].append((center_y, current_time))
        track_history[track_id]["last_seen"] = frame_idx
        center_y = smooth_position(track_history[track_id]["positions"])

        last_y = track_history[track_id]["last_y"]
        counted = track_history[track_id]["counted"]

        # Zone Logic
        zone = "middle"
        if center_y < ZONE_TOP:
            zone = "top"
        elif center_y > ZONE_BOTTOM:
            zone = "bottom"
        
        track_history[track_id]["zone_history"].append(zone)
        
        # Crossing logic
        zones = track_history[track_id]["zone_history"]
        if not counted and len(zones) >= 2:
            if "top" in zones and "bottom" in zones:
                if zones[0] == "top":
                    count_in += 1
                elif zones[0] == "bottom":
                    count_out += 1
                track_history[track_id]["counted"] = True
        track_history[track_id]["last_y"] = center_y

        # Speed calculation
        track = track_history[track_id]

        if len(track["positions"]) >= 2:
            y1, t1 = track["positions"][0]
            y2, t2 = track["positions"][-1]

            dy = y2 - y1
            dt = t2 - t1

            if abs(dy) > MIN_PIXEL_MOVE and dt > 0:
                speed_mps = (abs(dy) * PIXEL_TO_METER) / dt
                current_speed = speed_mps * 3.6
                
                if track["speed"] == 0.0:
                    track["speed"] = current_speed
                else:
                    alpha = 0.2  
                    track["speed"] = alpha * current_speed + (1 - alpha) * track["speed"]

            # Anomaly detection (speed-based)
            if track["speed"] > MAX_SPEED:
                track["anomaly"] = "overspeeding"
            elif track["speed"] < MIN_SPEED:
                if current_time - track["last_move_time"] > STOP_TIME_THRESHOLD:
                    track["anomaly"] = "stopped"
            else:
                track["anomaly"] = None
                track["last_move_time"] = current_time

        # direction determination
        if dy > 0:
            current_dir = "down"
        elif dy < 0:
            current_dir = "up"
        else:
            current_dir = None

        # anomaly direction
        if track["direction"] and current_dir != track["direction"]:
            track["anomaly"] = "WRONG_DIRECTION"

        track["direction"] = current_dir

        colors = []
        for tid in detections.tracker_id:
            anomaly = track_history.get(tid, {}).get("anomaly")
            if anomaly == "overspeeding":
                colors.append((0, 0, 255))
            elif anomaly == "stopped":
                colors.append((0, 255, 0))
            elif anomaly == "WRONG_DIRECTION":
                colors.append((255, 0, 0))
            else:
                colors.append((0, 255, 255))

    # Memory cleanup (hapus ID yang sudah lama hilang)
    if frame_idx % 30 == 0:
        ids_to_remove = [tid for tid, data in track_history.items() if frame_idx - data["last_seen"] > MAX_TRACK_AGE]
        for tid in ids_to_remove:
            del track_history[tid]

        

    # Draw line
    cv2.line(frame, (0, LINE_Y), (frame.shape[1], LINE_Y), (0, 0, 255), 2) 

    labels = []
    for track_id in detections.tracker_id:
        speed = track_history[track_id]["speed"]
        anomaly = track_history[track_id]["anomaly"]
        label = f"ID {track_id} | {speed:.1f} km/h"
        if anomaly:
            label += f" | {anomaly}"
        labels.append(label)

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