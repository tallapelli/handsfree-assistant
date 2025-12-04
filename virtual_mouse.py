import cv2
import mediapipe as mp
import pyautogui
import math
import time
import numpy as np
from collections import deque
import threading

class GestureController:
    def __init__(self):
        # Initialize MediaPipe Hands with improved settings
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.8,
            min_tracking_confidence=0.7
        )
        self.mp_draw = mp.solutions.drawing_utils
        
        # Screen and frame dimensions
        self.screen_width, self.screen_height = pyautogui.size()
        self.frame_width, self.frame_height = 640, 480
        
        # Improved smoothing with moving average
        self.position_history = deque(maxlen=3)  # Reduced for more responsive cursor
        
        # Enhanced gesture detection with better thresholds
        self.gesture_cooldown = 0.4  # Longer cooldown for better accuracy
        self.last_gesture_time = 0
        self.gesture_threshold_frames = 4  # More frames required for stability
        self.gesture_counters = {
            'cursor': 0,
            'left_click': 0,
            'right_click': 0,
            'scroll': 0
        }
        
        # More precise distance thresholds with better separation
        self.thresholds = {
            'cursor_control': 0.06,    # Index-thumb for cursor (back to original)
            'left_click': 0.04,        # Middle-thumb for left click (tighter)
            'right_click': 0.04,       # Little-thumb for right click (tighter)
            'scroll': 0.05             # Ring-thumb for scroll
        }
        
        # Coordinate mapping
        self.margin = 80
        self.dead_zone = 0.02
        
        # Scroll state management
        self.scroll_active = False
        self.scroll_reference_y = 0
        self.scroll_sensitivity = 0.02
        self.scroll_step = 3
        
        # Performance optimization
        pyautogui.FAILSAFE = False
        pyautogui.PAUSE = 0
        
        # Status tracking
        self.current_mode = "IDLE"
        self.fps_counter = 0
        self.fps_time = time.time()
        
        # Stop event for threading
        self.stop_event = None
        
        # Gesture state tracking for better accuracy
        self.last_distances = {
            'cursor': float('inf'),
            'left_click': float('inf'),
            'right_click': float('inf'),
            'scroll': float('inf')
        }
        
    def calculate_distance(self, p1, p2):
        """Calculate Euclidean distance between two landmarks."""
        return math.sqrt((p1.x - p2.x)**2 + (p1.y - p2.y)**2)
    
    def calculate_finger_angle(self, landmarks, finger_indices):
        """Calculate the angle of finger extension for better gesture detection."""
        # Get the three points of the finger (MCP, PIP, TIP)
        mcp = landmarks[finger_indices[0]]
        pip = landmarks[finger_indices[1]] 
        tip = landmarks[finger_indices[2]]
        
        # Calculate vectors
        v1 = np.array([pip.x - mcp.x, pip.y - mcp.y])
        v2 = np.array([tip.x - pip.x, tip.y - pip.y])
        
        # Calculate angle between vectors
        cosine_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
        cosine_angle = np.clip(cosine_angle, -1.0, 1.0)
        angle = np.arccos(cosine_angle)
        
        return np.degrees(angle)
    
    def is_finger_extended(self, landmarks, finger_indices, threshold_angle=160):
        """Check if a finger is extended based on joint angles."""
        try:
            angle = self.calculate_finger_angle(landmarks, finger_indices)
            return angle > threshold_angle
        except:
            return False
    
    def smooth_position(self, x, y):
        """Apply smoothing to cursor position using moving average."""
        self.position_history.append((x, y))
        
        if len(self.position_history) < 2:
            return x, y
        
        total_weight = 0
        smooth_x = 0
        smooth_y = 0
        
        for i, (px, py) in enumerate(self.position_history):
            weight = (i + 1) / len(self.position_history)
            smooth_x += px * weight
            smooth_y += py * weight
            total_weight += weight
        
        return int(smooth_x / total_weight), int(smooth_y / total_weight)
    
    def map_coordinates(self, hand_x, hand_y):
        """Map hand coordinates to screen coordinates with improved accuracy."""
        active_x_min = self.margin
        active_x_max = self.frame_width - self.margin
        active_y_min = self.margin
        active_y_max = self.frame_height - self.margin
        
        frame_x = hand_x * self.frame_width
        frame_y = hand_y * self.frame_height
        
        screen_x = np.interp(frame_x, [active_x_min, active_x_max], [0, self.screen_width])
        screen_y = np.interp(frame_y, [active_y_min, active_y_max], [0, self.screen_height])
        
        screen_x = max(0, min(self.screen_width - 1, screen_x))
        screen_y = max(0, min(self.screen_height - 1, screen_y))
        
        return screen_x, screen_y
    
    def is_gesture_stable(self, gesture_name):
        """Check if gesture is stable across multiple frames."""
        self.gesture_counters[gesture_name] += 1
        return self.gesture_counters[gesture_name] >= self.gesture_threshold_frames
    
    def reset_gesture_counters(self, exclude=None):
        """Reset gesture counters, optionally excluding one."""
        for key in self.gesture_counters:
            if exclude and key == exclude:
                continue
            self.gesture_counters[key] = 0
    
    def can_perform_gesture(self):
        """Check if enough time has passed since last gesture."""
        return time.time() - self.last_gesture_time > self.gesture_cooldown
    
    def perform_gesture(self, gesture_name):
        """Execute a gesture with proper timing."""
        if self.can_perform_gesture():
            self.last_gesture_time = time.time()
            return True
        return False
    
    def draw_debug_info(self, img, landmarks):
        """Draw debug information on the image."""
        if landmarks:
            thumb_tip = landmarks[4]
            index_tip = landmarks[8]
            middle_tip = landmarks[12]
            ring_tip = landmarks[16]
            little_tip = landmarks[20]
            
            # Calculate distances
            distances = {
                'cursor': self.calculate_distance(index_tip, thumb_tip),
                'left_click': self.calculate_distance(middle_tip, thumb_tip),
                'right_click': self.calculate_distance(little_tip, thumb_tip),
                'scroll': self.calculate_distance(ring_tip, thumb_tip)
            }
            
            # Draw active area rectangle
            cv2.rectangle(img, 
                         (self.margin, self.margin), 
                         (self.frame_width - self.margin, self.frame_height - self.margin),
                         (0, 255, 0), 2)
            
            # Draw distance indicators with better color coding
            y_offset = 30
            for gesture, distance in distances.items():
                threshold = self.thresholds.get(gesture.replace('_click', '').replace('_', ''), 0.05)
                
                # Color coding: Green = Active, Yellow = Close, White = Inactive
                if distance < threshold:
                    color = (0, 255, 0)  # Green - Active
                elif distance < threshold + 0.02:
                    color = (0, 255, 255)  # Yellow - Close
                else:
                    color = (255, 255, 255)  # White - Inactive
                    
                cv2.putText(img, f"{gesture}: {distance:.3f} ({'ACTIVE' if distance < threshold else 'inactive'})", 
                           (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
                y_offset += 20
            
            # Show current mode with larger text
            cv2.putText(img, f"Mode: {self.current_mode}", 
                       (10, self.frame_height - 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
            
            # Show gesture counters for debugging
            cv2.putText(img, f"Counters: L:{self.gesture_counters['left_click']} R:{self.gesture_counters['right_click']}", 
                       (10, self.frame_height - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
    
    def process_gestures(self, landmarks):
        """Process all hand gestures with improved accuracy and priority system."""
        if not landmarks:
            self.reset_gesture_counters()
            self.current_mode = "IDLE"
            return
        
        # Get landmark positions
        thumb_tip = landmarks[4]
        index_tip = landmarks[8]
        middle_tip = landmarks[12]
        ring_tip = landmarks[16]
        little_tip = landmarks[20]
        
        # Calculate distances
        cursor_distance = self.calculate_distance(index_tip, thumb_tip)
        left_click_distance = self.calculate_distance(middle_tip, thumb_tip)
        right_click_distance = self.calculate_distance(little_tip, thumb_tip)
        scroll_distance = self.calculate_distance(ring_tip, thumb_tip)
        
        # Store current distances
        self.last_distances['cursor'] = cursor_distance
        self.last_distances['left_click'] = left_click_distance
        self.last_distances['right_click'] = right_click_distance
        self.last_distances['scroll'] = scroll_distance
        
        # Check finger extension states for better accuracy
        index_extended = self.is_finger_extended(landmarks, [5, 6, 8])  # Index finger joints
        middle_extended = self.is_finger_extended(landmarks, [9, 10, 12])  # Middle finger joints
        ring_extended = self.is_finger_extended(landmarks, [13, 14, 16])  # Ring finger joints
        little_extended = self.is_finger_extended(landmarks, [17, 18, 20])  # Little finger joints
        
        # PRIORITY SYSTEM: Check click gestures FIRST (they have tighter thresholds)
        
        # --- LEFT CLICK (Middle finger + Thumb) - HIGHEST PRIORITY ---
        if (left_click_distance < self.thresholds['left_click'] and 
            not middle_extended and  # Middle finger should be bent/touching
            cursor_distance > self.thresholds['cursor_control']):  # Index should NOT be touching
            
            self.reset_gesture_counters(exclude='left_click')
            if self.is_gesture_stable('left_click') and self.perform_gesture('left_click'):
                self.current_mode = "LEFT_CLICK"
                pyautogui.click(button='left')
                print("✓ Left Click Performed")
                # Reset counters after successful click
                self.reset_gesture_counters()
                return  # Exit early to prevent other gestures
        
        # --- RIGHT CLICK (Little finger + Thumb) - HIGH PRIORITY ---
        elif (right_click_distance < self.thresholds['right_click'] and 
              not little_extended and  # Little finger should be bent/touching
              cursor_distance > self.thresholds['cursor_control']):  # Index should NOT be touching
            
            self.reset_gesture_counters(exclude='right_click')
            if self.is_gesture_stable('right_click') and self.perform_gesture('right_click'):
                self.current_mode = "RIGHT_CLICK"
                pyautogui.click(button='right')
                print("✓ Right Click Performed")
                # Reset counters after successful click
                self.reset_gesture_counters()
                return  # Exit early to prevent other gestures
        
        # --- SCROLL (Ring finger + Thumb) - MEDIUM PRIORITY ---
        elif (scroll_distance < self.thresholds['scroll'] and 
              not ring_extended and  # Ring finger should be bent/touching
              cursor_distance > self.thresholds['cursor_control']):  # Index should NOT be touching
            
            self.current_mode = "SCROLL"
            self.reset_gesture_counters(exclude='scroll')
            
            if not self.scroll_active:
                self.scroll_active = True
                self.scroll_reference_y = ring_tip.y
                print("🔄 Scroll Mode Activated")
            else:
                delta_y = ring_tip.y - self.scroll_reference_y
                
                if abs(delta_y) > self.scroll_sensitivity:
                    if delta_y < 0:  # Moving up
                        pyautogui.scroll(self.scroll_step)
                        print("⬆️ Scroll Up")
                    else:  # Moving down
                        pyautogui.scroll(-self.scroll_step)
                        print("⬇️ Scroll Down")
                    
                    self.scroll_reference_y = ring_tip.y
            return  # Exit early to prevent cursor control
        
        # --- CURSOR CONTROL (Index finger + Thumb) - LOWEST PRIORITY ---
        elif (cursor_distance < self.thresholds['cursor_control'] and
              left_click_distance > self.thresholds['left_click'] and  # Other fingers should NOT be touching
              right_click_distance > self.thresholds['right_click'] and
              scroll_distance > self.thresholds['scroll']):
            
            self.current_mode = "CURSOR"
            self.reset_gesture_counters(exclude='cursor')
            
            # Map hand position to screen coordinates
            screen_x, screen_y = self.map_coordinates(index_tip.x, index_tip.y)
            
            # Apply smoothing
            smooth_x, smooth_y = self.smooth_position(screen_x, screen_y)
            
            # Apply dead zone to prevent jittery movement
            current_x, current_y = pyautogui.position()
            if abs(smooth_x - current_x) > 3 or abs(smooth_y - current_y) > 3:
                pyautogui.moveTo(smooth_x, smooth_y, duration=0.005)
        
        else:
            # No gesture detected - reset states
            if self.scroll_active:
                self.scroll_active = False
                print("🔄 Scroll Mode Deactivated")
            
            self.reset_gesture_counters()
            self.current_mode = "IDLE"
    
    def update_fps(self):
        """Update FPS counter."""
        current_time = time.time()
        if current_time - self.fps_time >= 1.0:
            self.fps_time = current_time
            self.fps_counter = 0
        else:
            self.fps_counter += 1
    
    def run(self, stop_event=None):
        """Main execution loop with optional stop event for threading."""
        self.stop_event = stop_event
        
        print("🚀 Starting Enhanced Gesture Control...")
        print("📋 Gestures:")
        print("   👆 Index + Thumb (index extended) = Cursor Control")
        print("   🖕 Middle + Thumb (middle bent) = Left Click")
        print("   🤟 Little + Thumb (little bent) = Right Click")
        print("   💍 Ring + Thumb (ring bent) = Scroll")
        print("   ❌ Press 'q' to quit")
        print("🔧 Tips:")
        print("   - For clicks: Bend the finger to touch thumb")
        print("   - For cursor: Keep index extended while touching thumb")
        print("   - Ensure other fingers are away from thumb")
        print("=" * 60)
        
        # Initialize camera
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.frame_width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.frame_height)
        cap.set(cv2.CAP_PROP_FPS, 30)
        
        if not cap.isOpened():
            print("❌ Error: Could not open camera")
            return
        
        try:
            while True:
                if self.stop_event and self.stop_event.is_set():
                    print("🛑 Stop event received")
                    break
                
                success, img = cap.read()
                if not success:
                    print("⚠️ Warning: Failed to read from camera")
                    continue
                
                img = cv2.flip(img, 1)
                img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                results = self.hands.process(img_rgb)
                
                if results.multi_hand_landmarks:
                    for hand_landmarks in results.multi_hand_landmarks:
                        self.mp_draw.draw_landmarks(
                            img, hand_landmarks, self.mp_hands.HAND_CONNECTIONS,
                            self.mp_draw.DrawingSpec(color=(0, 255, 0), thickness=2),
                            self.mp_draw.DrawingSpec(color=(255, 0, 0), thickness=2)
                        )
                        
                        self.process_gestures(hand_landmarks.landmark)
                        self.draw_debug_info(img, hand_landmarks.landmark)
                else:
                    self.reset_gesture_counters()
                    self.current_mode = "NO_HAND_DETECTED"
                    if self.scroll_active:
                        self.scroll_active = False
                
                self.update_fps()
                cv2.imshow("Enhanced AI Virtual Mouse", img)
                
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                    
        except KeyboardInterrupt:
            print("\n🛑 Interrupted by user")
        except Exception as e:
            print(f"❌ Error: {e}")
        finally:
            cap.release()
            cv2.destroyAllWindows()
            print("✅ Gesture control stopped")

if __name__ == "__main__":
    controller = GestureController()
    controller.run()

def run_virtual_mouse(stop_event):
    """Function to run virtual mouse with stop event support."""
    controller = GestureController()
    controller.run(stop_event)