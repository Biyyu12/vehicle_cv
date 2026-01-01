import cv2
from ultralytics import YOLO

model = YOLO('yolov8n.pt')

cap = cv2.VideoCapture("assets/traffic.mp4")

tick_freq = cv2.getTickFrequency()
prev_tick = cv2.getTickCount()

# 2=car, 3=motorcycle, 5=bus, 7=truck
VEHICLE_CLASSES = [2, 3, 5, 7]

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    current_tick = cv2.getTickCount()
    time_diff = (current_tick - prev_tick) / tick_freq
    fps = 1 / time_diff if time_diff > 0 else 0
    prev_tick = current_tick

    results = model(frame, conf=0.5)

    for r in results:
        for box in r.boxes:
            cls_id = int(box.cls[0])
            if cls_id in VEHICLE_CLASSES:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = box.conf[0]

                label = f"{model.names[cls_id]} {conf:.2f}"

                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    cv2.putText(frame, f"FPS: {int(fps)}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.imshow("Vehicle Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
        
