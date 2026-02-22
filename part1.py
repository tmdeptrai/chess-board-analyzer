import cv2
import numpy as np
import json

# Read input from video
cap = cv2.VideoCapture("./input_videos/echiquier_vide_1.avi")
if not cap.isOpened():
    print("Video not found.")
    print("Have you tried to install the videos first? Run chmod +x ./download_videos.sh && ./download_videos.sh") 
    print("Or if you're a professor reading this, please copy all your .avi videos into the ./input_videos folder. Thank you :)") 
    
# Info for video writer
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))

fourcc = cv2.VideoWriter_fourcc(*'XVID') 
# Writer for the edges (Original resolution)
out_edges = cv2.VideoWriter('output_videos/part1_edges.avi', fourcc, fps, (frame_width, frame_height))
# Writer for the rectified board (400x400 resolution)
out_rectified = cv2.VideoWriter('output_videos/part1_rectified.avi', fourcc, fps, (400, 400))


def preprocess_frame(frame):
    """
    Preprocessing a frame:
    - Convert to gray
    - Gaussian blur to remove noises
    - A canny filter to extract edges
    - Finally perform a closure to fill gaps / connecting disconnected grid lines
    """
    gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray,(3,3),0)
    edges = cv2.Canny(blurred,50,150)
    solid_edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    return solid_edges
    
def get_grid_corners(edges):
    """
    Finds the 4 corners of the largest connected contour (the 8x8 grid)
    and ignoring all disconnected background noise (wood, letters, tripod).
    """
    # Find all contours in the binary edge map
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        return None
        
    # Sort contours by area to find the absolute largest one.
    # The 8x8 grid will have a massively larger area than the tripod or letters.
    largest_contour = max(contours, key=cv2.contourArea)
    
    # Extract the (x, y) coordinates of the pixels of this 8x8 grid
    points = largest_contour.reshape(-1, 2)
    
    # Apply the coordinate extremes math to the grid's pixels
    sums = points.sum(axis=1)
    diffs = np.diff(points, axis=1)
    
    top_left = points[np.argmin(sums)]
    bottom_right = points[np.argmax(sums)]
    top_right = points[np.argmin(diffs)]
    bottom_left = points[np.argmax(diffs)]
    
    return np.array([top_left, top_right, bottom_right, bottom_left], dtype=np.float32)

def warp_board(frame, corners, output_size=400):
    """
    Projects the distorted board into a perfect 2D square.
    
    (0,0)----------(400,0)
      |               |
      |               |
      |               |
      |               |
      |               |
    (0,400)--------(400,400)
    
    """
    dst_points = np.array([
        [0, 0], 
        [output_size, 0], 
        [output_size, output_size], 
        [0, output_size]
    ], dtype=np.float32)
    
    # Calculate Homography matrix
    matrix = cv2.getPerspectiveTransform(corners, dst_points)
    
    # Warp the image
    warped = cv2.warpPerspective(frame, matrix, (output_size, output_size))
    return warped

def put_text_with_outline(img, text):
    """Draw a visible text in an image."""
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 1
    thickness_out = 4
    thickness_in = 2
    
    height, width = img.shape[:2]
    text_size, _ = cv2.getTextSize(text, font, scale, thickness_out)
    text_w, text_h = text_size
    
    x = (width - text_w) // 2  
    y = height - 20
    pos = (x, y)
    
    cv2.putText(img, text, pos, font, scale, (0, 0, 0), thickness_out, cv2.LINE_AA)
    cv2.putText(img, text, pos, font, scale, (0, 255, 255), thickness_in, cv2.LINE_AA)

def create_dashboard(img1, img2, img3, img4, size=(400, 400)):
    """
    Resizes, annotates, and stacks 4 images into a 2x2 grid.
    """
    # Resize images
    i1 = cv2.resize(img1, size)
    i2 = cv2.resize(img2, size)
    i3 = cv2.resize(img3, size)
    i4 = cv2.resize(img4, size)
    
    # Add the labels
    put_text_with_outline(i1, "1. Original Frame")
    put_text_with_outline(i2, "2. Edges Detection")
    put_text_with_outline(i3, "3. Corners Extraction")
    put_text_with_outline(i4, "4. Rectified Board")
    
    # Stack into a grid
    top_row = np.hstack((i1, i2))
    bottom_row = np.hstack((i3, i4))
    grid = np.vstack((top_row, bottom_row))
    
    return grid
 
# ================= Main program =====================

corners = None

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("Video finished. Press any key in the OpenCV window to close...")
        cv2.waitKey(0)
        break    
    
    orig_frame = frame.copy()
    edges = preprocess_frame(frame)
    # cv2.imshow("Input video",frame)
    # cv2.imshow("Preprocessed video",edges)
    
    edges_bgr = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
    out_edges.write(edges_bgr)
    
    # Extract corners of the 8x8 grid
    if corners is None: #This will be calculated for ONLY 1 TIME
        corners = get_grid_corners(edges)
        with open('./corners_coordinates.json','w') as f:
            json.dump(corners.tolist(),f,indent=4)
        print("SUCCESS: Corners coordinates saved to part1_corners_coordinates.json!")    
    
    if corners is not None:
        # Draw the detected corners
        for point in corners:
            x, y = int(point[0]), int(point[1])
            cv2.circle(frame, (x, y), 5, (255, 255, 255), -1)
            
        # Warp the board
        warped_board = warp_board(orig_frame, corners)
        # cv2.imshow("Original with Corners", frame)
        # cv2.imshow("Rectified Board", warped_board)
        out_rectified.write(warped_board)
        
        dashboard = create_dashboard(orig_frame, edges_bgr, frame, warped_board)
        cv2.imshow("Chess Vision Pipeline Dashboard", dashboard)
    else:
        cv2.imshow("Original Image", frame)
        
    key = cv2.waitKey(1)
    if key == 27:  # esc key
        break

#Save the dashboard to an image (pretty cool for the report)
cv2.imwrite("./figures/part1_summary.png",dashboard)    
cap.release()
out_rectified.release()
out_edges.release()
cv2.destroyAllWindows()