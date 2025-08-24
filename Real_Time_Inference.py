import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import joblib
import json
import math
import time
from checkRules import EXERCISE_RULES

try:
    gru_model = load_model('files/GRU_model_7/GRU_model_7.h5')
    scaler = joblib.load('files/GRU_model_7/real_time_scaler.pkl')
    with open('files/GRU_model_7/encoded_labels.json', 'r') as f:
        label_mapping = json.load(f)
        label_mapping = {int(k): v for k, v in label_mapping.items()}
    print("GRU model and preprocessing files loaded successfully!")
except FileNotFoundError as e:
    print(f"Error: {e}. Make sure you have downloaded the required files (gru_model.h5, real_time_scaler.pkl, encoded_labels.json) and placed them in the same directory.")
    exit()
except Exception as e:
    print(f"An unexpected error occurred: {e}")
    exit()

No_Ex_Classes = ["Sitting", "Walking", "Standing Still"]

# Increased confidence thresholds for better stability
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(
    min_detection_confidence=0.7, 
    min_tracking_confidence=0.7,
    model_complexity=1  # Balanced complexity
)

POSE_LANDMARKS = {
    "left_shoulder": 11, "right_shoulder": 12,
    "left_elbow": 13, "right_elbow": 14,
    "left_wrist": 15, "right_wrist": 16,
    "left_hip": 23, "right_hip": 24,
    "left_knee": 25, "right_knee": 26,
    "left_ankle": 27, "right_ankle": 28,
    "nose": 0
}

# Include all angles with positive importance (21 features total)
ANGLE_DEFINITIONS = {
    "left_elbow_angle": ("left_shoulder", "left_elbow", "left_wrist"),
    "right_elbow_angle": ("right_shoulder", "right_elbow", "right_wrist"),
    "left_knee_angle": ("left_hip", "left_knee", "left_ankle"),
    "right_knee_angle": ("right_hip", "right_knee", "right_ankle"),
    "right_hip_angle": ("right_shoulder", "right_hip", "right_knee"),
    "shoulder_angle": ("left_elbow", "left_shoulder", "right_shoulder"),
    "hip_angle": ("left_knee", "left_hip", "right_hip"),
    "torso_angle": ("left_shoulder", "left_hip", "left_knee")
}

# Include all distances with positive importance
DISTANCE_DEFINITIONS = {
    "right_hand_to_right_hip_dist": ("right_wrist", "right_hip"),
    "left_hand_to_left_hip_dist": ("left_wrist", "left_hip"),
    "feet_distance": ("left_ankle", "right_ankle"),
}

# --- Helper functions ---
def calculate_angle(a, b, c):
    a, b, c = np.array(a), np.array(b), np.array(c)
    ba = a - b
    bc = c - b
    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-6)
    return np.degrees(np.arccos(np.clip(cosine_angle, -1.0, 1.0)))

def calculate_distance(p1, p2):
    return math.sqrt((p2[0] - p1[0])**2 + (p2[1] - p1[1])**2)

def calculate_vertical_angle(p1, p2):
    dx, dy = p2[0] - p1[0], p2[1] - p1[1]
    if dy == 0:
        return 90.0
    return math.degrees(math.atan(abs(dx) / abs(dy)))

def calculate_symmetry(l, r):
    if l == 0 and r == 0:
        return 1.0
    return 1.0 - abs(l - r) / max(l, r, 1e-6)

# --- Build full feature list (angles, distances + new engineered ones) ---
BASE_FEATURES = list(ANGLE_DEFINITIONS.keys()) + list(DISTANCE_DEFINITIONS.keys())

EXTRA_FEATURES = [
    "elbow_symmetry", "knee_symmetry",
    "left_hand_to_shoulder_vertical", "right_hand_to_shoulder_vertical",
    "left_elbow_to_hip_vertical", "right_elbow_to_hip_vertical",
    "left_ankle_to_knee_horizontal", "right_ankle_to_knee_horizontal",
    "torso_lean", "stance_width_ratio"
]
feature_names = BASE_FEATURES + EXTRA_FEATURES

sequence_length = 25
frame_sequence = []
prediction_label = "No Detection"
confidence = 0.0
no_person_label = "No Person Detected"
CONFIDENCE_THRESHOLD = 0.7  # Slightly lowered for better detection

last_prediction_time = time.time()
prediction_interval = 0.5

# Initialize exercise states
exercise_states = {ex: "start" for ex in EXERCISE_RULES.keys()}
exercise_reps = {ex: 0 for ex in EXERCISE_RULES.keys()}
posture_correctness = {ex: True for ex in EXERCISE_RULES.keys()}

cap = cv2.VideoCapture(0)
# Set camera resolution for better framing
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

window_name = 'Exercise Detector'
cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
cv2.resizeWindow(window_name, 1280, 720)

# Add frame counter for stabilization
frame_counter = 0
stable_landmarks = None

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    frame = cv2.flip(frame, 1)
    
    # Add a visual guide for proper framing
    h, w, _ = frame.shape
    cv2.rectangle(frame, (w//4, h//6), (3*w//4, 5*h//6), (0, 255, 0), 2)
    cv2.putText(frame, "Stand in this area", (w//4+10, h//6-10), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    image.flags.writeable = False
    results = pose.process(image)
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    
    # --- CHECK FOR PERSON DETECTION AND DRAW LANDMARKS ---
    if results.pose_landmarks:
        # Simple stabilization: average landmarks over frames
        frame_counter += 1
        current_landmarks = results.pose_landmarks.landmark
        
        if stable_landmarks is None:
            # Initialize stable_landmarks with current values
            stable_landmarks = []
            for lm in current_landmarks:
                stable_landmarks.append({
                    'x': lm.x,
                    'y': lm.y,
                    'z': lm.z,
                    'visibility': lm.visibility
                })
        else:
            # Weighted average for stabilization
            for i in range(len(current_landmarks)):
                stable_landmarks[i]['x'] = 0.7 * stable_landmarks[i]['x'] + 0.3 * current_landmarks[i].x
                stable_landmarks[i]['y'] = 0.7 * stable_landmarks[i]['y'] + 0.3 * current_landmarks[i].y
                stable_landmarks[i]['z'] = 0.7 * stable_landmarks[i]['z'] + 0.3 * current_landmarks[i].z
                stable_landmarks[i]['visibility'] = 0.7 * stable_landmarks[i]['visibility'] + 0.3 * current_landmarks[i].visibility
        
        # Draw landmarks with stabilization
        # Create a copy of the original landmarks for drawing
        drawing_landmarks = results.pose_landmarks
        for i, lm in enumerate(drawing_landmarks.landmark):
            lm.x = stable_landmarks[i]['x']
            lm.y = stable_landmarks[i]['y']
            lm.z = stable_landmarks[i]['z']
            lm.visibility = stable_landmarks[i]['visibility']
            
        mp.solutions.drawing_utils.draw_landmarks(image, drawing_landmarks, mp_pose.POSE_CONNECTIONS)
        
        current_features = {}
        try:
            # Use stabilized landmarks for feature calculation
            keypoints = {}
            for name, idx in POSE_LANDMARKS.items():
                keypoints[name] = [stable_landmarks[idx]['x'], stable_landmarks[idx]['y']]

            # 1. Base angles for the MODEL (21 features)
            for angle_name, (a, b, c) in ANGLE_DEFINITIONS.items():
                current_features[angle_name] = calculate_angle(keypoints[a], keypoints[b], keypoints[c])
            
            # 2. Base distances for the MODEL
            for dist_name, (p1, p2) in DISTANCE_DEFINITIONS.items():
                current_features[dist_name] = calculate_distance(keypoints[p1], keypoints[p2])

            # 3. All engineered features for the MODEL
            current_features["elbow_symmetry"] = calculate_symmetry(current_features["left_elbow_angle"], current_features["right_elbow_angle"])
            current_features["knee_symmetry"] = calculate_symmetry(current_features["left_knee_angle"], current_features["right_knee_angle"])
            current_features["left_hand_to_shoulder_vertical"] = keypoints["left_shoulder"][1] - keypoints["left_wrist"][1]
            current_features["right_hand_to_shoulder_vertical"] = keypoints["right_shoulder"][1] - keypoints["right_wrist"][1]
            current_features["left_elbow_to_hip_vertical"] = keypoints["left_hip"][1] - keypoints["left_elbow"][1]
            current_features["right_elbow_to_hip_vertical"] = keypoints["right_hip"][1] - keypoints["right_elbow"][1]
            current_features["left_ankle_to_knee_horizontal"] = abs(keypoints["left_ankle"][0] - keypoints["left_knee"][0])
            current_features["right_ankle_to_knee_horizontal"] = abs(keypoints["right_ankle"][0] - keypoints["right_knee"][0])
            current_features["torso_lean"] = calculate_vertical_angle(keypoints["left_shoulder"], keypoints["left_hip"])
            shoulder_width = calculate_distance(keypoints["left_shoulder"], keypoints["right_shoulder"])
            current_features["stance_width_ratio"] = current_features["feet_distance"] / shoulder_width if shoulder_width > 0 else 0

            # --- Calculate Rule-Based Angles SEPARATELY ---
            # These are for the checkRules logic only, not included in the model's feature vector
            angles_for_rules = {}
            angles_for_rules["left_shoulder_angle"] = calculate_angle(keypoints["left_hip"], keypoints["left_shoulder"], keypoints["left_elbow"])
            angles_for_rules["right_shoulder_angle"] = calculate_angle(keypoints["right_hip"], keypoints["right_shoulder"], keypoints["right_elbow"])
            angles_for_rules["left_elbow_angle"] = current_features["left_elbow_angle"]
            angles_for_rules["right_elbow_angle"] = current_features["right_elbow_angle"]
            angles_for_rules["left_knee_angle"] = current_features["left_knee_angle"]
            angles_for_rules["right_knee_angle"] = current_features["right_knee_angle"]
            
            # Calculate left_hip_angle for rules only (not part of the 21 features)
            angles_for_rules["left_hip_angle"] = calculate_angle(keypoints["left_shoulder"], keypoints["left_hip"], keypoints["left_knee"])
            angles_for_rules["right_hip_angle"] = current_features["right_hip_angle"]
            angles_for_rules["torso_angle"] = current_features["torso_angle"]

        except Exception as e:
            # fallback if something breaks
            print(f"Error during feature calculation: {e}")
            for col in feature_names:
                current_features[col] = 0
            angles_for_rules = {}
            for key in ["left_shoulder_angle", "right_shoulder_angle", "left_elbow_angle", "right_elbow_angle", 
                       "left_knee_angle", "right_knee_angle", "left_hip_angle", "right_hip_angle", "torso_angle"]:
                angles_for_rules[key] = 0

        # Convert into feature vector with correct order (21 features only)
        feature_vector = np.array([current_features[key] for key in feature_names], dtype=np.float32)
        frame_sequence.append(feature_vector)
        frame_sequence = frame_sequence[-sequence_length:]
    
        current_time = time.time()
    
        if len(frame_sequence) == sequence_length and (current_time - last_prediction_time) >= prediction_interval:
            sequence_to_scale = np.array(frame_sequence).reshape(-1, len(feature_names))
            scaled_sequence = scaler.transform(sequence_to_scale)
            input_data = scaled_sequence.reshape(1, sequence_length, len(feature_names))
            
            prediction_probabilities = gru_model.predict(input_data, verbose=0)[0]
            predicted_class_index = np.argmax(prediction_probabilities)
            confidence = prediction_probabilities[predicted_class_index]
            
            if confidence >= CONFIDENCE_THRESHOLD:
                predicted_label = label_mapping[predicted_class_index]
                if predicted_label not in No_Ex_Classes:
                    prediction_label = predicted_label
                    if prediction_label in EXERCISE_RULES:
                        rule_fn = EXERCISE_RULES[prediction_label]
                        posture, new_state, new_reps = rule_fn(angles_for_rules,
                                                               exercise_states[prediction_label],
                                                               exercise_reps[prediction_label])
                        exercise_states[prediction_label] = new_state
                        exercise_reps[prediction_label] = new_reps
                        posture_correctness[prediction_label] = posture
                else:
                    prediction_label = "No Exercise"
            else:
                prediction_label = "No Exercise" 
            
            last_prediction_time = current_time
                
    else:
        prediction_label = no_person_label
        confidence = 0.0
        frame_sequence = [] 
        stable_landmarks = None  # Reset stabilization when no person detected
        
    # --- Display Logic ---
    h, w, _ = image.shape
    text_size = cv2.FONT_HERSHEY_SIMPLEX
    text_scale = 0.8
    text_thickness = 2
    padding = 15

    # Define a background for better text visibility
    background_rect_height = 100
    cv2.rectangle(image, (0, 0), (w, background_rect_height), (30, 30, 30), -1)

    # Place text logically on separate lines or with sufficient padding
    text_y_pos = 30
    
    # Line 1: Exercise and Confidence
    text_exercise = f'Exercise: {prediction_label}'
    cv2.putText(image, text_exercise, (padding, text_y_pos), 
                text_size, text_scale, (0, 255, 0) if confidence > CONFIDENCE_THRESHOLD else (0, 0, 255), text_thickness)
                
    text_confidence = f'Confidence: {confidence:.2f}'
    text_w, _ = cv2.getTextSize(text_exercise, text_size, text_scale, text_thickness)[0]
    cv2.putText(image, text_confidence, (padding + text_w + padding, text_y_pos), 
                text_size, text_scale, (255, 255, 0), text_thickness)
    
    text_y_pos += 40

    # Line 2: Reps and Form
    if prediction_label in EXERCISE_RULES and prediction_label not in No_Ex_Classes:
        reps = exercise_reps.get(prediction_label, 0)
        posture = posture_correctness.get(prediction_label, True)
        
        text_reps = f'Reps: {reps}'
        cv2.putText(image, text_reps, (padding, text_y_pos), 
                    text_size, text_scale, (0, 255, 0), text_thickness)

        text_form = f'Form: {"OK" if posture else "Fix"}'
        text_w, _ = cv2.getTextSize(text_reps, text_size, text_scale, text_thickness)[0]
        cv2.putText(image, text_form, (padding + text_w + padding, text_y_pos), 
                    text_size, text_scale, (0, 255, 0) if posture else (0, 0, 255), text_thickness) 

    cv2.imshow(window_name, image)

    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()