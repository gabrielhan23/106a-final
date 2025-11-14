import cv2
import numpy as np

# --- You can tune these parameters ---

# Rectangle (Device) Parameters
MIN_RECT_AREA = 5000       # Min area for the "device"
RECT_EPSILON = 0.03        # How "perfect" the rectangle must be (lower is stricter)

# Oval (Port) Parameters
MIN_OVAL_AREA = 1000       # Min area for the "port"
MIN_OVAL_RATIO = 0.3       # 1.0 is a circle, 0.1 is a line
MAX_OVAL_RATIO = 0.9       # Filter out perfect circles

# General Parameters
CANNY_LOW = 50
CANNY_HIGH = 150

# --- End of tuneable parameters ---

# Start the webcam
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # 1. Pre-processing: Grayscale and Blur
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)

    # 2. Edge Detection
    edges = cv2.Canny(blur, CANNY_LOW, CANNY_HIGH)

    # 3. Find All Contours
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # 4. Filter Contours into two lists: rectangles and ovals
    detected_rectangles = []
    detected_ovals = []

    for contour in contours:
        area = cv2.contourArea(contour)

        # --- Check for Rectangles (Devices) ---
        if area > MIN_RECT_AREA:
            # Approximate the contour to a polygon
            perimeter = cv2.arcLength(contour, True)
            # This 'epsilon' value is key for rectangle detection
            epsilon = RECT_EPSILON * perimeter
            approx_corners = cv2.approxPolyDP(contour, epsilon, True)

            # A rectangle has 4 corners
            if len(approx_corners) == 4 and cv2.isContourConvex(approx_corners):
                detected_rectangles.append(approx_corners)

        # --- Check for Ovals (Ports) ---
        if area > MIN_OVAL_AREA and len(contour) >= 5:
            try:
                ellipse = cv2.fitEllipse(contour)
                (cx, cy), (minor_axis, major_axis), angle = ellipse

                if minor_axis > 0 and major_axis > 0:
                    ratio = minor_axis / major_axis
                    if MIN_OVAL_RATIO < ratio < MAX_OVAL_RATIO:
                        # Store the oval's center point and its bounding box
                        oval_center = (int(cx), int(cy))
                        oval_bbox = cv2.boundingRect(contour)
                        detected_ovals.append((oval_center, oval_bbox))
            except cv2.error:
                pass # fitEllipse can fail on strange shapes

    # 5. The "Model" Logic: Check if ovals are inside rectangles
    for oval_center, oval_bbox in detected_ovals:
        is_on_device = False
        
        for rect_contour in detected_rectangles:
            # Check if the oval's center is inside the rectangle's contour
            # This is the core logical check
            result = cv2.pointPolygonTest(rect_contour, oval_center, False)
            
            if result >= 0: # >= 0 means inside or on the edge
                is_on_device = True
                
                # (Optional) Draw the "device" rectangle in red
                cv2.drawContours(frame, [rect_contour], 0, (0, 0, 255), 2)
                break # Found its parent device, no need to check others

        # Only draw the green box if the oval is confirmed to be on a device
        if is_on_device:
            x, y, w, h = oval_bbox
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(frame, "VALID PORT", (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # (Optional) Draw the oval's center
            cv2.circle(frame, oval_center, 5, (255, 0, 0), -1)


    # Display the resulting frame
    cv2.imshow('Validated Port Detector', frame)
    cv2.imshow('Edges', edges) # For tuning

    if cv2.waitKey(1) == ord('q'):
        break

# Clean up
cap.release()
cv2.destroyAllWindows()