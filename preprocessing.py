import cv2
import numpy as np
import os

# 1. Setup paths
video_path = "./input_videos/echiquier_vide_1.avi"
output_path = "./figures/preprocessing_workflow.jpg"

# Ensure output directory exists
os.makedirs(os.path.dirname(output_path), exist_ok=True)

cap = cv2.VideoCapture(video_path)
ret, frame = cap.read()
cap.release()

if not ret:
    print("Error: Could not read frame from video.")
    exit()

# 2. Preprocessing Steps
gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
blurred = cv2.GaussianBlur(gray, (3, 3), 0)
edges = cv2.Canny(blurred, 50, 150)
closure = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))

# 3. Convert grayscale results to BGR for stacking
# We use the 'frame' for original, and convert others
res1 = frame.copy()
res2 = cv2.cvtColor(blurred, cv2.COLOR_GRAY2BGR)
res3 = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
res4 = cv2.cvtColor(closure, cv2.COLOR_GRAY2BGR)

# 4. Add Labels
font = cv2.FONT_HERSHEY_SIMPLEX
images = [res1, res2, res3, res4]
titles = ["Original", "Gaussian Blur", "Canny Edges Detection", "Morphological Closure"]

for img, title in zip(images, titles):
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 2
    thickness_out = 8
    thickness_in = 4
    
    height, width = img.shape[:2]
    text_size, _ = cv2.getTextSize(title, font, scale, thickness_out)
    text_w, text_h = text_size
    
    x = (width - text_w) // 2  
    y = height - 20
    pos = (x, y)
    
    cv2.putText(img, title, pos, font, scale, (0, 0, 0), thickness_out, cv2.LINE_AA)
    cv2.putText(img, title, pos, font, scale, (0, 255, 255), thickness_in, cv2.LINE_AA)


# 5. Create Horizontal Dashboard and Save
dashboard = np.hstack(images)
cv2.imwrite(output_path, dashboard)
print(f"Dashboard saved successfully at: {output_path}")