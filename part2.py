import cv2
import numpy as np
import json
import time

def load_coordinates(coordinates_file = 'corners_coordinates.json'):
    try:
        with open(coordinates_file,'r') as file:
            coordinates = json.load(file)
        return np.array(coordinates,dtype=np.float32)            
    except Exception as e:
        print(f"Error loading coordinates: ",e)
        return None

def put_text_with_outline(img, text, pos):
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 1
    thickness_out = 4
    thickness_in = 2
    cv2.putText(img, text, pos, font, scale, (0, 0, 0), thickness_out, cv2.LINE_AA)
    cv2.putText(img, text, pos, font, scale, (0, 255, 255), thickness_in, cv2.LINE_AA)

def create_part2_dashboard(img1, img2, img3, size=(400, 400)):
    i1 = cv2.resize(img1, size)
    i2 = cv2.resize(img2, size)
    i3 = cv2.resize(img3, size)
    
    put_text_with_outline(i1, "1. Last Stable Board", (20, size[1] - 20))
    put_text_with_outline(i2, "2. Motion Mask", (20, size[1] - 20))
    put_text_with_outline(i3, "3. Hand Contour", (20, size[1] - 20))
    
    dashboard = np.hstack((i1, i2, i3))
    return dashboard

class ChessHandMovementDetector:
    def __init__(self):
        self.coordinates = load_coordinates()
        
        #Get the rectified perspective
        self.width, self.height = 800, 800
        self.dst_points = np.array([[0, 0], [self.width - 1, 0], 
                                    [self.width - 1, self.height - 1], 
                                    [0, self.height - 1]], dtype="float32")
        if self.coordinates is not None:
            self.matrix = cv2.getPerspectiveTransform(self.coordinates, self.dst_points)
        else:
            self.matrix = None
            
        self.backSub = cv2.createBackgroundSubtractorMOG2(history=100, varThreshold=50, detectShadows=False)
        self.hand_is_present = False
        self.last_stable_board = None
        self.figure_saved = False
        self.hand_detected_frames = 0
    
    def detect_hands(self,frame):
        fgMask = self.backSub.apply(frame)
        contours, _ = cv2.findContours(fgMask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        min_area = 1000
        hand_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > min_area]

        return hand_contours, fgMask
    
    def process_frame(self,frame):
        if self.matrix is None:
            board = frame
        else:
            board = cv2.warpPerspective(frame, self.matrix, (self.width, self.height))

        debug_img = board.copy()
        hand_contours, fgMask = self.detect_hands(board)
        currently_detected = len(hand_contours) > 0

        if currently_detected and not self.hand_is_present:
            print("EVENT: Hand has entered the board.")
            self.hand_is_present = True
            self.last_stable_board = board.copy()

        elif not currently_detected and self.hand_is_present:
            print("EVENT: Hand has left the board. A move was likely made.")
            self.hand_is_present = False
            self.hand_detected_frames = 0
            print("Analyzing the move...")
            
        if self.hand_is_present:
            self.hand_detected_frames += 1
            cv2.putText(debug_img, "Hand Detected", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            cv2.drawContours(debug_img, hand_contours, -1, (0, 255, 0), 2)
            
            if not self.figure_saved and self.last_stable_board is not None and self.hand_detected_frames > 5:
                mask_bgr = cv2.cvtColor(fgMask, cv2.COLOR_GRAY2BGR)
                dashboard = create_part2_dashboard(self.last_stable_board, mask_bgr, debug_img)
                cv2.imwrite("figures/part2_summary.png", dashboard)
                print("SUCCESS: Saved summary figure to figures/part2_summary.png")
                self.figure_saved = True
        else:
            cv2.putText(debug_img, "Board is Stable", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                
        return debug_img
                
                
# ===== MAIN WORFLOW ======

detector = ChessHandMovementDetector()

cap = cv2.VideoCapture("input_videos/mat_du_lion_1.avi")

# Video Writer
fps = int(cap.get(cv2.CAP_PROP_FPS))
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out_video = cv2.VideoWriter('output_videos/part2_hand_detection.avi', fourcc, fps, (detector.width, detector.height))


while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("Video finished. Press any key in the OpenCV window to close...")
        cv2.waitKey(0)
        break
    
    cv2.imshow("Original",frame)
    result_frame = detector.process_frame(frame)
    cv2.imshow("Processed Frame",result_frame)
    
    out_video.write(result_frame)
    
    key = cv2.waitKey(1)
    if key == 27:  # esc key
        break

cap.release()
out_video.release()
cv2.destroyAllWindows()