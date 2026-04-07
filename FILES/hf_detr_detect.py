import cv2
from transformers import pipeline
from PIL import Image
import numpy as np
import time

# --- 1. CONFIGURATION ---
VIDEO_PATH = 0
OUTPUT_PATH = 'output_results/detr_outputs/detr_output.avi'

# We use the standard DETR model from Facebook (Meta)
# It's highly accurate but might be slower on CPU than YOLOv5n
MODEL_NAME = "facebook/detr-resnet-50"

# --- 2. LOAD HUGGING FACE PIPELINE ---
print(f"[INFO] Downloading/Loading model: {MODEL_NAME}...")
# This 'pipeline' command does ALL the hard work for you
# It downloads the model, the processor, and sets up the AI
detector = pipeline("object-detection", model=MODEL_NAME)

# --- 3. VIDEO SETUP ---
cap = cv2.VideoCapture(VIDEO_PATH)

if not cap.isOpened():
    print(f"[ERROR] Cannot open video: {VIDEO_PATH}")
    exit()

# Get video properties
frame_width = int(cap.get(3))
frame_height = int(cap.get(4))
fps = int(cap.get(cv2.CAP_PROP_FPS))

# Use MJPG/AVI for safety on your laptop
out = cv2.VideoWriter(OUTPUT_PATH, cv2.VideoWriter_fourcc(*'MJPG'), fps, (frame_width, frame_height))

print(f"[INFO] Starting inference on {VIDEO_PATH}...")
print("[INFO] Press 'q' to stop early.")

# --- 4. PROCESSING LOOP ---
start_time = time.time()
frame_count = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1

    # CONVERSION STEP: OpenCV uses BGR, but Hugging Face needs RGB PIL Images
    # 1. Convert BGR to RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    # 2. Convert to PIL Image
    pil_image = Image.fromarray(rgb_frame)

    # --- AI INFERENCE ---
    # Pass the PIL image to the pipeline
    results = detector(pil_image)

    # --- DRAWING RESULTS ---
    for result in results:
        # DETR returns [xmin, ymin, xmax, ymax]
        box = result['box']
        label = result['label']
        score = result['score']

        # Only draw if confident (e.g., > 50%)
        if score > 0.5:
            xmin, ymin = int(box['xmin']), int(box['ymin'])
            xmax, ymax = int(box['xmax']), int(box['ymax'])

            # Draw Box (Green)
            cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)

            # Draw Label
            label_text = f"{label}: {score:.2f}"
            cv2.putText(frame, label_text, (xmin, ymin - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Write and Show
    out.write(frame)
    cv2.imshow('Hugging Face DETR', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# --- CLEANUP ---
end_time = time.time()
print(f"\n[INFO] Finished! Processed {frame_count} frames in {end_time - start_time:.2f} seconds.")
cap.release()
out.release()
cv2.destroyAllWindows()
