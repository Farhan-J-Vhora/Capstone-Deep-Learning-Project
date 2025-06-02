import cv2
import numpy as np
import math

class AdvancedFingerCounter:
    def __init__(self):
        # ROI coordinates
        self.roi_x = 100
        self.roi_y = 100  
        self.roi_width = 300
        self.roi_height = 300
        
        # Skin detection parameters
        self.lower_skin_hsv = np.array([0, 20, 70])
        self.upper_skin_hsv = np.array([20, 255, 255])
        
        # Alternative skin range
        self.lower_skin_ycrcb = np.array([0, 133, 77])
        self.upper_skin_ycrcb = np.array([255, 173, 127])
        
    def detect_skin_hsv(self, frame):
        """Detect skin using HSV color space"""
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, self.lower_skin_hsv, self.upper_skin_hsv)
        return mask
    
    def detect_skin_ycrcb(self, frame):
        """Detect skin using YCrCb color space (often better for skin)"""
        ycrcb = cv2.cvtColor(frame, cv2.COLOR_BGR2YCrCb)
        mask = cv2.inRange(ycrcb, self.lower_skin_ycrcb, self.upper_skin_ycrcb)
        return mask
    
    def preprocess_mask(self, mask):
        """Clean up the mask using morphological operations"""
        # Remove noise with opening
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=2)
        
        # Fill holes with closing
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (8, 8))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
        
        # Smooth the mask
        mask = cv2.GaussianBlur(mask, (5, 5), 0)
        
        return mask
    
    def find_hand_contour(self, mask):
        """Find the largest contour which should be the hand"""
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return None
        
        # Find the largest contour
        largest_contour = max(contours, key=cv2.contourArea)
        
        # Check if contour is large enough
        if cv2.contourArea(largest_contour) < 5000:
            return None
        
        # Smooth the contour
        epsilon = 0.005 * cv2.arcLength(largest_contour, True)
        smoothed_contour = cv2.approxPolyDP(largest_contour, epsilon, True)
        
        return smoothed_contour
    
    def count_fingers_advanced(self, contour):
        """Advanced finger counting using multiple techniques"""
        try:
            # Method 1: Convexity defects
            hull_indices = cv2.convexHull(contour, returnPoints=False)
            
            if len(hull_indices) < 4:
                return 0
            
            defects = cv2.convexityDefects(contour, hull_indices)
            
            if defects is None:
                return 0
            
            finger_count = 0
            
            for i in range(defects.shape[0]):
                s, e, f, d = defects[i, 0]
                start = tuple(contour[s][0])
                end = tuple(contour[e][0])
                far = tuple(contour[f][0])
                
                # Calculate angle between the lines from defect point to start and end points
                a = np.linalg.norm(np.array(start) - np.array(far))
                b = np.linalg.norm(np.array(end) - np.array(far))
                c = np.linalg.norm(np.array(start) - np.array(end))
                
                if b == 0 or a == 0:
                    continue
                
                # Calculate angle using cosine rule
                angle = math.acos((a**2 + b**2 - c**2) / (2 * a * b))
                angle_deg = math.degrees(angle)
                
                # Only count as finger if angle is acute and defect is deep enough
                if angle_deg <= 90 and d > 15000:
                    finger_count += 1
            
            # Fingers = defects + 1 (but max 5)
            return min(finger_count + 1, 5)
            
        except Exception as e:
            return 0
    
    def draw_hand_info(self, frame, contour, finger_count):
        """Draw hand contour and finger count"""
        if contour is not None:
            # Draw contour
            cv2.drawContours(frame, [contour], -1, (0, 255, 0), 2)
            
            # Draw convex hull
            hull = cv2.convexHull(contour)
            cv2.drawContours(frame, [hull], -1, (255, 0, 0), 2)
            
            # Find and mark defects
            hull_indices = cv2.convexHull(contour, returnPoints=False)
            if len(hull_indices) > 3:
                defects = cv2.convexityDefects(contour, hull_indices)
                if defects is not None:
                    for i in range(defects.shape[0]):
                        s, e, f, d = defects[i, 0]
                        far = tuple(contour[f][0])
                        cv2.circle(frame, far, 8, (255, 255, 0), -1)
        
        # Draw finger count with large, clear text
        cv2.putText(frame, f'FINGERS: {finger_count}', (10, 50), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)
        
        return frame
    
    def run(self):
        """Main function"""
        cap = cv2.VideoCapture(0)
        
        if not cap.isOpened():
            print("Error: Cannot open camera")
            return
        
        print("=== ADVANCED FINGER COUNTER ===")
        print("Instructions:")
        print("1. Place your hand in the blue rectangle")
        print("2. Use good lighting")
        print("3. Keep background simple")
        print("4. Press 'q' to quit")
        print("5. Press 's' to switch skin detection method")
        
        use_ycrcb = False  # Start with HSV
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Flip frame horizontally for mirror effect
            frame = cv2.flip(frame, 1)
            
            # Draw ROI rectangle
            roi_start = (self.roi_x, self.roi_y)
            roi_end = (self.roi_x + self.roi_width, self.roi_y + self.roi_height)
            cv2.rectangle(frame, roi_start, roi_end, (255, 0, 0), 2)
            
            # Extract ROI
            roi = frame[self.roi_y:self.roi_y + self.roi_height, 
                       self.roi_x:self.roi_x + self.roi_width]
            
            # Detect skin
            if use_ycrcb:
                skin_mask = self.detect_skin_ycrcb(roi)
                method_text = "YCrCb"
            else:
                skin_mask = self.detect_skin_hsv(roi)
                method_text = "HSV"
            
            # Preprocess mask
            clean_mask = self.preprocess_mask(skin_mask)
            
            # Find hand contour
            hand_contour = self.find_hand_contour(clean_mask)
            
            # Count fingers
            finger_count = 0
            if hand_contour is not None:
                finger_count = self.count_fingers_advanced(hand_contour)
                
                # Draw contour on ROI
                roi_with_contour = roi.copy()
                roi_with_contour = self.draw_hand_info(roi_with_contour, hand_contour, finger_count)
                frame[self.roi_y:self.roi_y + self.roi_height, 
                      self.roi_x:self.roi_x + self.roi_width] = roi_with_contour
            
            # Draw finger count on main frame
            cv2.putText(frame, f'FINGERS: {finger_count}', (50, 50), 
                       cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 4)
            
            # Show detection method
            cv2.putText(frame, f'Method: {method_text}', (50, 400), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            # Show instructions
            cv2.putText(frame, 'Press S to switch method, Q to quit', (50, 430), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            
            # Display frames
            cv2.imshow('Finger Counter', frame)
            
            # Show mask for debugging
            if clean_mask is not None:
                cv2.imshow('Skin Mask', clean_mask)
            
            # Handle key presses
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                use_ycrcb = not use_ycrcb
                print(f"Switched to {'YCrCb' if use_ycrcb else 'HSV'} skin detection")
        
        cap.release()
        cv2.destroyAllWindows()
        print("Finger counter stopped.")

if __name__ == "__main__":
    counter = AdvancedFingerCounter()
    counter.run()