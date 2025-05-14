import cv2
import mediapipe as mp
import pyautogui
import numpy as np
import time

# Initialize MediaPipe hands
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7,  # Increased for stability
    min_tracking_confidence=0.8    # Increased for stability
)

# Simplified drawing styles (only key points)
hand_landmark_style = mp_drawing.DrawingSpec(color=(0, 255, 128), thickness=2, circle_radius=3)

# Set up webcam with minimal resolution
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)  # Lower resolution to boost FPS
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)
screen_width, screen_height = pyautogui.size()
pyautogui.FAILSAFE = False

# Smoothing variables with enhanced approach
previous_smoothed_pos = None
smoothing_factor = 0.15  # Reduced for smoother movement
position_history = []
history_length = 5  # Number of frames for position averaging

# Mouse control states
click_state = False
click_hold_frames = 0
click_hold_threshold = 3  # Increased threshold
click_cooldown = 5  # Increased cooldown
click_cooldown_counter = 0
drag_state = False

def calculate_distance(p1, p2):
    """Calculate Euclidean distance between two points"""
    return np.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

def smooth_position(current_raw_pos, previous_smoothed_pos, position_history):
    """Enhanced position smoothing with acceleration handling"""
    if previous_smoothed_pos is None:
        return current_raw_pos
    
    # Add current position to history
    position_history.append(current_raw_pos)
    if len(position_history) > history_length:
        position_history.pop(0)
    
    # Calculate average position from history
    avg_x = sum(pos[0] for pos in position_history) / len(position_history)
    avg_y = sum(pos[1] for pos in position_history) / len(position_history)
    
    # Calculate velocity
    velocity_x = current_raw_pos[0] - previous_smoothed_pos[0]
    velocity_y = current_raw_pos[1] - previous_smoothed_pos[1]
    
    # Adjust smoothing factor based on velocity
    dynamic_smoothing = smoothing_factor * (1 + 0.5 * (abs(velocity_x) + abs(velocity_y)))
    dynamic_smoothing = min(0.5, max(0.1, dynamic_smoothing))
    
    # Apply smoothing
    smoothed_pos = (
        dynamic_smoothing * avg_x + (1 - dynamic_smoothing) * previous_smoothed_pos[0],
        dynamic_smoothing * avg_y + (1 - dynamic_smoothing) * previous_smoothed_pos[1]
    )
    return smoothed_pos

def calculate_stable_pinch(index_pos, thumb_pos, hand_size, history_length=3):
    """Calculate pinch distance with temporal stability"""
    distance = calculate_distance(index_pos, thumb_pos)
    normalized_distance = distance / hand_size
    return normalized_distance

# Minimal instructions
print("\n=== Hand Gesture Mouse ===")
print("• Move: Index finger")
print("• Click/Drag: Pinch")
print("• Exit: 'q'")
print("==================\n")

start_time = time.time()
frame_count = 0

while cap.isOpened():
    success, image = cap.read()
    if not success:
        print("Camera frame failed.")
        continue

    # Flip image
    image = cv2.flip(image, 1)
    image_height, image_width, _ = image.shape
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = hands.process(image_rgb)
    
    # Update cooldown
    click_cooldown_counter = max(0, click_cooldown_counter - 1)
    
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Draw only key landmarks
            for idx in [mp_hands.HandLandmark.INDEX_FINGER_TIP, 
                       mp_hands.HandLandmark.THUMB_TIP, 
                       mp_hands.HandLandmark.WRIST, 
                       mp_hands.HandLandmark.MIDDLE_FINGER_MCP]:
                landmark = hand_landmarks.landmark[idx]
                x, y = int(landmark.x * image_width), int(landmark.y * image_height)
                cv2.circle(image, (x, y), 3, (0, 255, 128), -1)
            
            # Get key landmarks
            index_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
            thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
            wrist = hand_landmarks.landmark[mp_hands.HandLandmark.WRIST]
            middle_mcp = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_MCP]
            
            # Convert to pixel coordinates
            x = int(index_tip.x * image_width)
            y = int(index_tip.y * image_height)
            thumb_x = int(thumb_tip.x * image_width)
            thumb_y = int(thumb_tip.y * image_height)
            wrist_x = int(wrist.x * image_width)
            wrist_y = int(wrist.y * image_height)
            middle_x = int(middle_mcp.x * image_width)
            middle_y = int(middle_mcp.y * image_height)
            
            # Calculate hand size and padding
            hand_size = calculate_distance((wrist_x, wrist_y), (middle_x, middle_y))
            padding = int(hand_size * 0.6)
            
            # Map to screen coordinates
            mouse_x = np.interp(x, [padding, image_width - padding], [0, screen_width])
            mouse_y = np.interp(y, [padding, image_height - padding], [0, screen_height])
            
            # Move mouse with enhanced smoothing
            current_raw_pos = (mouse_x, mouse_y)
            smoothed_pos = smooth_position(current_raw_pos, previous_smoothed_pos, position_history)
            previous_smoothed_pos = smoothed_pos
            
            # Add deadzone for small movements
            movement_threshold = 2
            if abs(smoothed_pos[0] - pyautogui.position()[0]) > movement_threshold or \
               abs(smoothed_pos[1] - pyautogui.position()[1]) > movement_threshold:
                smooth_x = max(0, min(int(smoothed_pos[0]), screen_width - 1))
                smooth_y = max(0, min(int(smoothed_pos[1]), screen_height - 1))
                pyautogui.moveTo(smooth_x, smooth_y)
            
            # Enhanced pinch detection
            pinch_distance = calculate_stable_pinch(
                (x, y), (thumb_x, thumb_y), hand_size
            )
            pinch_threshold = 0.3  # Normalized threshold
            
            # Simplified pinch visualization
            mid_x, mid_y = (x + thumb_x) // 2, (y + thumb_y) // 2
            pinch_radius = int(max(8, 20 - pinch_distance * 0.2))
            color = (0, 255, 0) if pinch_distance < pinch_threshold else (0, 128, 255)
            cv2.circle(image, (mid_x, mid_y), pinch_radius, color, 1)
            
            # Click/drag logic
            click_state = pinch_distance < pinch_threshold
            if click_state:
                click_hold_frames += 1
                if click_hold_frames >= click_hold_threshold and click_cooldown_counter == 0:
                    if not drag_state:
                        pyautogui.mouseDown()
                        drag_state = True
            else:
                if drag_state:
                    pyautogui.mouseUp()
                    drag_state = False
                    click_cooldown_counter = click_cooldown
                click_hold_frames = 0
    
    # Minimal FPS display
    frame_count += 1
    elapsed_time = time.time() - start_time
    fps = frame_count / elapsed_time if elapsed_time > 0 else 0
    cv2.putText(image, f"FPS: {fps:.1f}", (10, 20), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    
    # Display frame
    cv2.imshow('Hand Gesture Mouse', image)
    
    # Exit on 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
