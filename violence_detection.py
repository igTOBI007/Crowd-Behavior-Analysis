import cv2
import numpy as np
from collections import deque

class ViolenceDetector:
    """
    Violence detection using optical flow analysis.
    """
    
    def __init__(self, 
                 flow_threshold=2.5,
                 violence_threshold=0.35,
                 history_size=15):
        
        self.flow_threshold = flow_threshold
        self.violence_threshold = violence_threshold
        self.history_size = history_size
        
        self.flow_history = deque(maxlen=history_size)
        self.prev_gray = None
        self.violence_detected = False
        
    def calculate_optical_flow(self, current_frame, prev_frame):
        """Calculate dense optical flow between frames."""
        gray_current = cv2.cvtColor(current_frame, cv2.COLOR_BGR2GRAY)
        gray_prev = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
        
        flow = cv2.calcOpticalFlowFarneback(
            gray_prev, gray_current,
            None,
            pyr_scale=0.5,
            levels=3,
            winsize=15,
            iterations=3,
            poly_n=5,
            poly_sigma=1.2,
            flags=0
        )
        
        magnitude, angle = cv2.cartToPolar(flow[..., 0], flow[..., 1])
        
        return magnitude, angle, flow
    
    def analyze_flow_chaos(self, magnitude, angle):
        """Analyze flow patterns for chaotic/violent behavior."""
        avg_magnitude = np.mean(magnitude)
        max_magnitude = np.max(magnitude)
        
        high_flow_ratio = np.sum(magnitude > self.flow_threshold) / magnitude.size
        
        angle_std = np.std(angle[magnitude > self.flow_threshold]) if np.any(magnitude > self.flow_threshold) else 0
        angle_variance = angle_std / np.pi
        
        chaos_score = (high_flow_ratio * 0.4) + (avg_magnitude / 20.0 * 0.3) + (angle_variance * 0.3)
        chaos_score = min(chaos_score, 1.0)
        
        return chaos_score, avg_magnitude, high_flow_ratio
    
    def detect_violence(self, frame, detections=None, prev_detections=None):
        """Main violence detection function."""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        if self.prev_gray is None:
            self.prev_gray = gray
            return False, 0.0, 0.0
        
        prev_frame_bgr = cv2.cvtColor(self.prev_gray, cv2.COLOR_GRAY2BGR) if len(self.prev_gray.shape) == 2 else self.prev_gray
        
        magnitude, angle, flow = self.calculate_optical_flow(frame, prev_frame_bgr)
        
        chaos_score, avg_mag, high_flow_ratio = self.analyze_flow_chaos(magnitude, angle)
        
        self.flow_history.append(chaos_score)
        
        if len(self.flow_history) > 5:
            recent_avg = np.mean(list(self.flow_history)[-5:])
            violence_level = recent_avg
        else:
            violence_level = chaos_score
        
        is_violent = violence_level > self.violence_threshold
        
        self.prev_gray = gray
        self.violence_detected = is_violent
        
        return is_violent, violence_level, avg_mag
    
    def draw_violence_indicator(self, frame, is_violent, violence_level):
        """Draw violence detection indicator on frame."""
        h, w = frame.shape[:2]
        
        if is_violent:
            color = (0, 0, 255)
            status = "VIOLENCE DETECTED!"
        elif violence_level > 0.25:
            color = (0, 165, 255)
            status = "HIGH ACTIVITY"
        else:
            color = (0, 255, 0)
            status = "NORMAL"
        
        cv2.rectangle(frame, (0, 0), (w, 70), (0, 0, 0), -1)
        cv2.rectangle(frame, (0, 0), (w, 70), color, 3)
        
        cv2.putText(frame, status, (20, 50), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1.5, color, 3)
        
        bar_length = int((w - 40) * min(violence_level, 1.0))
        cv2.rectangle(frame, (20, h - 50), (w - 20, h - 20), (50, 50, 50), -1)
        cv2.rectangle(frame, (20, h - 50), (20 + bar_length, h - 20), color, -1)
        
        level_text = f"Violence Level: {violence_level:.2%}"
        cv2.putText(frame, level_text, (25, h - 60), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        return frame