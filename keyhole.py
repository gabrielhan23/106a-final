import cv2
import numpy as np

# === Configuration ===
# Search region: (x, y, width, height) or None for full image
SEARCH_REGION = (1200, 1200, 1500, 1500)

# Match threshold: 0.0 to 1.0 (higher = stricter)
MATCH_THRESHOLD = 0.6

# Template scale range as percentage of search area
MIN_TEMPLATE_SIZE = 0.001  # 0.1% of search area
MAX_TEMPLATE_SIZE = 0.02   # 2% of search area
NUM_SCALES = 20

# === 1. Load Images ===
image = cv2.imread("close.jpeg")
template = cv2.imread("usbc_template.png")

if image is None or template is None:
    print("Error: Could not load images.")
    exit()

# Convert to grayscale (preserves light/dark contrast)
image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
template_gray = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)

# === 2. Setup Search Region ===
if SEARCH_REGION is not None:
    roi_x, roi_y, roi_w, roi_h = SEARCH_REGION
    roi_x = max(0, roi_x)
    roi_y = max(0, roi_y)
    roi_w = min(roi_w, image_gray.shape[1] - roi_x)
    roi_h = min(roi_h, image_gray.shape[0] - roi_y)
    search_gray = image_gray[roi_y:roi_y+roi_h, roi_x:roi_x+roi_w]
    search_offset = (roi_x, roi_y)
    print(f"Searching region: ({roi_x}, {roi_y}) size {roi_w}x{roi_h}")
else:
    search_gray = image_gray
    search_offset = (0, 0)
    print("Searching entire image")

# Store all matches
all_matches = []

# === 3. Single-Scale Template Matching with Correlation Map ===
tH, tW = template_gray.shape[:2]
search_area = search_gray.shape[0] * search_gray.shape[1]
template_area = tH * tW

# Calculate scale range
min_scale = np.sqrt((MIN_TEMPLATE_SIZE * search_area) / template_area)
max_scale = np.sqrt((MAX_TEMPLATE_SIZE * search_area) / template_area)

# Use middle scale for visualization
display_scale = (min_scale + max_scale) / 2

print(f"Template area: {template_area} pixels")
print(f"Displaying correlation map at scale {display_scale:.3f}")

# Resize template to display scale
resized_template = cv2.resize(template_gray, None, fx=display_scale, fy=display_scale, interpolation=cv2.INTER_AREA)
rH, rW = resized_template.shape[:2]

# Perform template matching at this scale
correlation_map = cv2.matchTemplate(search_gray, resized_template, cv2.TM_CCOEFF_NORMED)

print(f"Correlation map range: {correlation_map.min():.3f} to {correlation_map.max():.3f}")

# Create heatmap visualization
# Normalize correlation map to 0-255 for display
correlation_normalized = ((correlation_map - correlation_map.min()) / 
                          (correlation_map.max() - correlation_map.min()) * 255).astype(np.uint8)

# Apply colormap (hot = red/yellow for high correlation, blue/black for low)
heatmap = cv2.applyColorMap(correlation_normalized, cv2.COLORMAP_JET)

# Resize heatmap to match search region size (correlation map is smaller)
heatmap_resized = cv2.resize(heatmap, (search_gray.shape[1], search_gray.shape[0]), interpolation=cv2.INTER_LINEAR)

# Create overlay on search region
search_region_color = cv2.cvtColor(search_gray, cv2.COLOR_GRAY2BGR)
overlay = cv2.addWeighted(search_region_color, 0.4, heatmap_resized, 0.6, 0)

# Mark threshold line on heatmap
threshold_locations = np.where(correlation_map >= MATCH_THRESHOLD)
for pt in zip(*threshold_locations[::-1]):
    cv2.circle(overlay, pt, 2, (255, 255, 255), -1)

print(f"Found {len(threshold_locations[0])} points above threshold {MATCH_THRESHOLD}")

# Now do multi-scale for finding all matches and best scale heatmap
print(f"\nSearching at scales {min_scale:.3f} to {max_scale:.3f}...")

best_correlation_map = None
best_scale_match = None
best_max_score = -1

for scale in np.linspace(min_scale, max_scale, NUM_SCALES)[::-1]:
    # Resize template
    resized = cv2.resize(template_gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    rH, rW = resized.shape[:2]
    
    # Skip if too large
    if rH > search_gray.shape[0] or rW > search_gray.shape[1]:
        continue
    
    # Template matching (TM_CCOEFF_NORMED is good for contrast matching)
    result = cv2.matchTemplate(search_gray, resized, cv2.TM_CCOEFF_NORMED)
    
    # Track best scale for heatmap
    max_val = result.max()
    if max_val > best_max_score:
        best_max_score = max_val
        best_correlation_map = result
        best_scale_match = scale
    
    # Find all matches above threshold
    locations = np.where(result >= MATCH_THRESHOLD)
    for pt in zip(*locations[::-1]):
        all_matches.append({
            'score': result[pt[1], pt[0]],
            'location': pt,
            'scale': scale,
            'size': (rW, rH)
        })

# === 4. Remove Overlapping Matches (Non-Maximum Suppression) ===
print(f"\nFound {len(all_matches)} raw matches above threshold {MATCH_THRESHOLD}")

if len(all_matches) == 0:
    print("No matches found. Try lowering MATCH_THRESHOLD.")
    exit()

# Sort by score
all_matches.sort(key=lambda x: x['score'], reverse=True)

# Simple NMS: remove overlapping boxes
def non_max_suppression(matches, overlap_thresh=0.3):
    if len(matches) == 0:
        return []
    
    boxes = []
    for m in matches:
        x, y = m['location']
        w, h = m['size']
        x += search_offset[0]
        y += search_offset[1]
        boxes.append([x, y, x + w, y + h, m['score']])
    
    boxes = np.array(boxes)
    x1, y1, x2, y2, scores = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3], boxes[:, 4]
    
    areas = (x2 - x1) * (y2 - y1)
    order = scores.argsort()[::-1]
    
    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        
        w = np.maximum(0, xx2 - xx1)
        h = np.maximum(0, yy2 - yy1)
        
        overlap = (w * h) / areas[order[1:]]
        order = order[np.concatenate([[0], np.where(overlap <= overlap_thresh)[0] + 1])[1:]]
    
    return [matches[i] for i in keep]

filtered_matches = non_max_suppression(all_matches, overlap_thresh=0.3)
print(f"After removing overlaps: {len(filtered_matches)} unique matches")

# === 5. Create Heatmap for Best Match Scale ===
print(f"\nBest match found at scale {best_scale_match:.3f} with score {best_max_score:.3f}")

# Create heatmap for the best scale
correlation_normalized = ((best_correlation_map - best_correlation_map.min()) / 
                          (best_correlation_map.max() - best_correlation_map.min()) * 255).astype(np.uint8)

heatmap_best = cv2.applyColorMap(correlation_normalized, cv2.COLORMAP_JET)
heatmap_best_resized = cv2.resize(heatmap_best, (search_gray.shape[1], search_gray.shape[0]), interpolation=cv2.INTER_LINEAR)

search_region_color = cv2.cvtColor(search_gray, cv2.COLOR_GRAY2BGR)
overlay_best = cv2.addWeighted(search_region_color, 0.4, heatmap_best_resized, 0.6, 0)

# Mark the best match location
best_loc = np.unravel_index(best_correlation_map.argmax(), best_correlation_map.shape)
cv2.circle(overlay_best, (best_loc[1], best_loc[0]), 10, (255, 255, 255), 3)
cv2.circle(overlay_best, (best_loc[1], best_loc[0]), 5, (0, 255, 0), -1)

# Mark all matches above threshold
threshold_locations_best = np.where(best_correlation_map >= MATCH_THRESHOLD)
for pt in zip(*threshold_locations_best[::-1]):
    cv2.circle(overlay_best, pt, 2, (255, 255, 255), -1)

# === 6. Draw Results ===
result_image = image.copy()

colors = [(0, 255, 0), (255, 0, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255), (0, 255, 255)]

for idx, match in enumerate(filtered_matches):
    x = match['location'][0] + search_offset[0]
    y = match['location'][1] + search_offset[1]
    w, h = match['size']
    
    color = colors[idx % len(colors)]
    cv2.rectangle(result_image, (x, y), (x + w, y + h), color, 3)
    
    label = f"#{idx+1}: {match['score']:.2f}"
    cv2.putText(result_image, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    
    print(f"Match #{idx+1}: Score={match['score']:.3f}, Scale={match['scale']:.2f}, Pos=({x},{y})")

# Draw search region
if SEARCH_REGION is not None:
    rx, ry, rw, rh = SEARCH_REGION
    cv2.rectangle(result_image, (rx, ry), (rx + rw, ry + rh), (255, 0, 0), 2)

# Display
cv2.imshow("1. Template", template_gray)
cv2.imshow("2. Heatmap at Best Match Scale", overlay_best)
cv2.imshow("3. Matches Found", result_image)

print("\nHeatmap (at best scale):")
print("  RED/YELLOW = High correlation (good match)")
print("  GREEN = Medium correlation")
print("  BLUE/BLACK = Low correlation (poor match)")
print("  WHITE DOTS = Above threshold")
print("  GREEN DOT WITH WHITE RING = Best match location")

cv2.waitKey(0)
cv2.destroyAllWindows()