"""
Main script with integrated violence detection.
"""

import cv2
import csv
import numpy as np
import winsound  # 🔔 NEW
from config import *
from violence_detection import ViolenceDetector
import time

print("Loading YOLO model...")
net = cv2.dnn.readNetFromDarknet(YOLO_CONFIG["CONFIG_PATH"], YOLO_CONFIG["WEIGHTS_PATH"])
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
ln = net.getLayerNames()
ln = [ln[i - 1] for i in net.getUnconnectedOutLayers()]

print("Initializing violence detector...")
violence_detector = ViolenceDetector(
    flow_threshold=2.5,
    violence_threshold=0.35,
    history_size=15
)

video_path = VIDEO_CONFIG["VIDEO_CAP"]
print(f"Opening video: {video_path}")
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("ERROR: Could not open video file!")
    exit()

fps = int(cap.get(cv2.CAP_PROP_FPS))
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

print(f"Video properties: {width}x{height} @ {fps} FPS, {total_frames} frames")

output_path = "output_with_violence_detection.mp4"
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
video_writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

frame_count = 0
violence_count = 0
total_people_detected = 0
prev_detections = None

# 🔔 ALERT CONTROL FLAG (NEW)
alert_triggered = False

start_time = time.time()

print("\nProcessing video with violence detection...")
print("Press 'q' to quit early\n")

# ================= CSV SETUP =================
file = open("report.csv", "w", newline="")
csv_writer = csv.writer(file)
csv_writer.writerow(["Frame", "People", "Violence"])

# ================= MAIN LOOP =================
while True:
    frame_count += 1

    ret, frame = cap.read()
    if not ret:
        break

    if FRAME_SIZE:
        h, w = frame.shape[:2]
        if h > FRAME_SIZE or w > FRAME_SIZE:
            scale = FRAME_SIZE / max(h, w)
            frame = cv2.resize(frame, (int(w * scale), int(h * scale)))

    blob = cv2.dnn.blobFromImage(frame, 1/255.0, (416, 416), swapRB=True, crop=False)
    net.setInput(blob)
    layerOutputs = net.forward(ln)

    boxes = []
    confidences = []
    H, W = frame.shape[:2]

    for output in layerOutputs:
        for detection in output:
            scores = detection[5:]
            classID = np.argmax(scores)
            confidence = scores[classID]

            if classID == 0 and confidence > MIN_CONF:
                box = detection[0:4] * np.array([W, H, W, H])
                (centerX, centerY, width_box, height_box) = box.astype("int")
                x = int(centerX - (width_box / 2))
                y = int(centerY - (height_box / 2))

                boxes.append([x, y, int(width_box), int(height_box)])
                confidences.append(float(confidence))

    idxs = cv2.dnn.NMSBoxes(boxes, confidences, MIN_CONF, NMS_THRESH)

    current_detections = []
    person_count = 0

    if len(idxs) > 0:
        for i in idxs.flatten():
            (x, y, w, h) = boxes[i]
            current_detections.append([x, y, w, h])

            if SHOW_DETECT:
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

            person_count += 1

    total_people_detected += person_count

    # ================= VIOLENCE DETECTION =================
    is_violent, violence_level, motion_intensity = violence_detector.detect_violence(
        frame, current_detections, prev_detections
    )

    if is_violent:
        violence_count += 1

        # ALERT TEXT
        cv2.putText(frame, "ALERT!", (50, 50),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1, (0, 0, 255), 3)

        # 🔔 SMART SOUND ALERT
        if not alert_triggered:
            print("⚠️ Violence Detected")
            winsound.Beep(1000, 500)
            alert_triggered = True
    else:
        alert_triggered = False

    frame = violence_detector.draw_violence_indicator(frame, is_violent, violence_level)

    # ================= DISPLAY =================
    cv2.putText(frame, f"People: {person_count}", (20, H - 70),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    # ================= CSV LOGGING =================
    csv_writer.writerow([frame_count, person_count, violence_level])

    # ================= SAVE VIDEO =================
    video_writer.write(frame)

    if SHOW_PROCESSING_OUTPUT:
        cv2.imshow('Crowd Analysis + Violence Detection', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("\nEarly exit requested")
            break

    if frame_count % 30 == 0:
        progress = (frame_count / total_frames) * 100 if total_frames > 0 else 0
        violence_status = "YES" if is_violent else "NO"
        print(f"Frame {frame_count}/{total_frames} ({progress:.1f}%) - "
              f"Violence: {violence_status} ({violence_level:.2%}) - "
              f"Total violent frames: {violence_count}")

    prev_detections = current_detections

# ================= CLEANUP =================
end_time = time.time()
elapsed = end_time - start_time

cap.release()
video_writer.release()
cv2.destroyAllWindows()
file.close()

# ================= SUMMARY =================
print("\n" + "="*60)
print("PROCESSING COMPLETE - RESULTS SUMMARY")
print("="*60)
print(f"Video: {video_path}")
print(f"Output saved to: {output_path}")
print(f"\nFRAME STATISTICS:")
print(f"  Total frames processed: {frame_count}")
print(f"  Processing time: {elapsed:.2f} seconds")
print(f"  Processing FPS: {frame_count/elapsed:.2f}")
print(f"\nCROWD DETECTION:")
print(f"  Total people detected: {total_people_detected}")
print(f"  Average people per frame: {total_people_detected/frame_count:.2f}")
print(f"\nVIOLENCE DETECTION:")
print(f"  Violence detected in: {violence_count} frames")
print(f"  Violence percentage: {(violence_count/frame_count)*100:.2f}%")
print(f"  Violence duration: {violence_count/fps:.2f} seconds")

if violence_count > 0:
    print(f"  ⚠️  WARNING: Violent behavior detected in video!")
else:
    print(f"  ✅  No violence detected - peaceful crowd behavior")

print("="*60)