import cv2
import torch
from ultralytics import FastSAM

# --- Input Mode ---
USE_IMAGE = False       # Set to True to use an image file, False to use webcam
IMAGE_PATH = "far.jpeg"  # Path to the image file (only used if USE_IMAGE is True)

# --- Parameters ---
MODEL_PATH = 'FastSAM-x.pt'  # Change to 'FastSAM-x.pt' to test the large model
IMAGE_SIZE = 640
CONFIDENCE = 0.5
IOU_THRESHOLD = 0.7
# --- End of Parameters ---

# 1. Determine the device (Auto-detects Apple Silicon 'mps' GPU)
if torch.backends.mps.is_available():
    DEVICE = 'mps'
elif torch.cuda.is_available():
    DEVICE = 'cuda'
else:
    DEVICE = 'cpu'

print(f"========================================")
print(f"Using device: {DEVICE}")
print(f"========================================")

# 2. Load the Fast-SAM Model
try:
    model = FastSAM(MODEL_PATH)
except FileNotFoundError:
    print(f"Error: Model file not found at {MODEL_PATH}")
    print("Please download the .pt file and place it in the same folder.")
    exit()
except Exception as e:
    print(f"Error loading model: {e}")
    exit()

# 3. Initialize video capture or load image
if USE_IMAGE:
    # Load image from file
    frame = cv2.imread(IMAGE_PATH)
    if frame is None:
        print(f"Error: Could not load image from {IMAGE_PATH}")
        exit()
    print(f"Loaded image: {IMAGE_PATH}")
    use_video = False
else:
    # Start the webcam
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        exit()
    use_video = True

while True:
    # Read a frame from the webcam or use the loaded image
    if use_video:
        ret, frame = cap.read()
        if not ret:
            break
    # If using image, frame is already loaded

    # 4. Run Fast-SAM inference on the frame
    results = model(frame,
                    device=DEVICE,
                    retina_masks=True,
                    imgsz=IMAGE_SIZE,
                    conf=CONFIDENCE,
                    iou=IOU_THRESHOLD,
                    verbose=False) # Set to True to see timing info

    # 5. Get the first result
    result = results[0]

    # 6. Use the built-in .plot() method to draw all masks and boxes
    # This is the easiest way to visualize the raw model output
    annotated_frame = result.plot()

    # Display the resulting frame
    cv2.imshow("Fast-SAM on MPS (Press 'q' to quit)", annotated_frame)

    # For images, wait indefinitely; for video, check every 1ms
    wait_time = 0 if not use_video else 1
    if cv2.waitKey(wait_time) == ord('q'):
        break

# Clean up
if use_video:
    cap.release()
cv2.destroyAllWindows()