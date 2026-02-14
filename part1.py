import cv2
import numpy as np
import os
os.makedirs('output_videos', exist_ok=True) #folder for output video


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

fourcc = cv2.VideoWriter_fourcc(*'XVID') #XVID codec, reliable for Linux
out = cv2.VideoWriter('output_videos/part1_edges.avi', fourcc, fps, (frame_width, frame_height))

def preprocess_frame(frame):
    """
    Preprocessing a frame:
    - Convert to gray
    - Gaussian blur to remove noises
    - Finally a canny filter to extract edges
    """
    gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray,(3,3),0)
    edges = cv2.Canny(blurred,50,150)
    return edges
    
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break    
    
    edges = preprocess_frame(frame)
    cv2.imshow("Input video",frame)
    cv2.imshow("Preprocessed video",edges)
    
    edges_bgr = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
    out.write(edges_bgr)
    
    key = cv2.waitKey(1)
    if key == 27:  # esc key
        break
    
cap.release()
out.release()
cv2.destroyAllWindows()