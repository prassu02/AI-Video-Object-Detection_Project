import cv2
import numpy as np
import time
import os
import sys

# --- 1. CONFIGURATION ---
PROTOTXT_PATH = 'mobilenet_files/MobileNetSSD_deploy.prototxt'
MODEL_PATH = 'mobilenet_files/MobileNetSSD_deploy.caffemodel'
VIDEO_PATH = 0
OUTPUT_PATH = 'output_results/mobilenet-outputs/mobilenet_output_bolts.avi'
CONFIDENCE_THRESHOLD = 0.3

# --- 2. SYSTEM CHECKS (Fixing the "Silent Failures") ---
# A. Check if input files actually exist
if not os.path.exists(PROTOTXT_PATH) or not os.path.exists(MODEL_PATH):
    print(f"[ERROR] Model files missing! Check 'mobilenet_files' folder.")
    sys.exit(1)
if not os.path.exists(VIDEO_PATH):
    print(f"[ERROR] Video file not found: {VIDEO_PATH}")
    sys.exit(1)

# B. Force-create the output directory (OpenCV cannot do this itself)
output_dir = os.path.dirname(OUTPUT_PATH)
if not os.path.exists(output_dir):
    print(f"[INFO] Creating missing folder: {output_dir}")
    os.makedirs(output_dir, exist_ok=True)

# --- 3. LOAD MODEL ---
print("[INFO] Loading MobileNet SSD...")
net = cv2.dnn.readNetFromCaffe(PROTOTXT_PATH, MODEL_PATH)

CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
           "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
           "dog", "horse", "motorbike", "person", "pottedplant", "sheep",
           "sofa", "train", "tvmonitor", "bolt", "book", "bagpack", "headphones", "mobile"]
COLORS = np.random.uniform(0, 255, size=(len(CLASSES), 3))

# --- 4. VIDEO SETUP ---
print(f"[INFO] Opening video: {VIDEO_PATH}")
cap = cv2.VideoCapture(VIDEO_PATH)

if not cap.isOpened():
    print("[ERROR] Cannot open video. Check path or codec.")
    sys.exit(1)

# C. Read the first frame to get the EXACT size (Crucial for VideoWriter)
ret, frame = cap.read()
if not ret:
    print("[ERROR] Video is empty or cannot be read.")
    sys.exit(1)

# Reset video to start
cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

frame_height, frame_width = frame.shape[:2]
fps = int(cap.get(cv2.CAP_PROP_FPS))
print(f"[INFO] Video loaded: {frame_width}x{frame_height} at {fps} FPS")

# Initialize Video Writer with MJPG (Safe Codec)
fourcc = cv2.VideoWriter_fourcc(*'MJPG')
out = cv2.VideoWriter(OUTPUT_PATH, fourcc, fps, (frame_width, frame_height))

# --- 5. PROCESSING LOOP ---
print("[INFO] Starting detection... Press 'q' to stop.")
start_time = time.time()
frame_count = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1

    # Resize frame to 300x300 for the model
    blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 0.007843, (300, 300), 127.5)
    net.setInput(blob)
    detections = net.forward()

    # Loop over detections
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]

        if confidence > CONFIDENCE_THRESHOLD:
            idx = int(detections[0, 0, i, 1])
            label = CLASSES[idx]

            # Scale box back to original frame size
            box = detections[0, 0, i, 3:7] * np.array([frame_width, frame_height, frame_width, frame_height])
            (startX, startY, endX, endY) = box.astype("int")

            # Draw
            label_text = "{}: {:.2f}%".format(label, confidence * 100)
            cv2.rectangle(frame, (startX, startY), (endX, endY), COLORS[idx], 2)
            y = startY - 15 if startY - 15 > 15 else startY + 15
            cv2.putText(frame, label_text, (startX, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS[idx], 2)

    # Show and Save
    cv2.imshow("MobileNet Detection", frame)
    out.write(frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# --- CLEANUP ---
end_time = time.time()
print(f"\n[INFO] Finished! Processed {frame_count} frames in {end_time - start_time:.2f} seconds.")
print(f"[INFO] Video saved successfully at: {OUTPUT_PATH}")

cap.release()
out.release()
cv2.destroyAllWindows()
