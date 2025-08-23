import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import joblib
import json
import math
import time

try:
    gru_model = load_model('files/GRU_model_6/GRU_model_6.h5')
    scaler = joblib.load('files/GRU_model_6/real_time_scaler.pkl')
    with open('files/GRU_model_6/encoded_labels.json', 'r') as f:
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

mp_pose = mp.solutions.pose
pose = mp_pose.Pose(min_detection_confidence=0.3, min_tracking_confidence=0.3)

POSE_LANDMARKS = {
    "left_shoulder": 11, "right_shoulder": 12,
    "left_elbow": 13, "right_elbow": 14,
    "left_wrist": 15, "right_wrist": 16,
    "left_hip": 23, "right_hip": 24,
    "left_knee": 25, "right_knee": 26,
    "left_ankle": 27, "right_ankle": 28,
    "nose": 0
}

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
CONFIDENCE_THRESHOLD = 0.8

last_prediction_time = time.time()
prediction_interval = 0.5

cap = cv2.VideoCapture(0)
window_name = 'Exercise Detector'
cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
cv2.resizeWindow(window_name, 1280, 720)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    frame = cv2.flip(frame, 1)

    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    image.flags.writeable = False
    results = pose.process(image)
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    
    # --- CHECK FOR PERSON DETECTION ---
    if results.pose_landmarks:
        current_features = {}
        try:
            landmarks = results.pose_landmarks.landmark
            keypoints = {name: [landmarks[idx].x, landmarks[idx].y] for name, idx in POSE_LANDMARKS.items()}

            # 1. Base angles
            for angle_name, (a, b, c) in ANGLE_DEFINITIONS.items():
                current_features[angle_name] = calculate_angle(keypoints[a], keypoints[b], keypoints[c])
            
            # Additional left_hip_angle as per your list
            current_features["left_hip_angle"] = calculate_angle(keypoints["left_shoulder"], keypoints["left_hip"], keypoints["left_knee"])

            # 2. Base distances
            for dist_name, (p1, p2) in DISTANCE_DEFINITIONS.items():
                current_features[dist_name] = calculate_distance(keypoints[p1], keypoints[p2])

            # 3. All engineered features (as per your list)
            # Symmetry features
            current_features["elbow_symmetry"] = calculate_symmetry(current_features["left_elbow_angle"], current_features["right_elbow_angle"])
            current_features["knee_symmetry"] = calculate_symmetry(current_features["left_knee_angle"], current_features["right_knee_angle"])
            
            # Hand vs shoulder
            current_features["left_hand_to_shoulder_vertical"] = keypoints["left_shoulder"][1] - keypoints["left_wrist"][1]
            current_features["right_hand_to_shoulder_vertical"] = keypoints["right_shoulder"][1] - keypoints["right_wrist"][1]

            # Elbow vs hip
            current_features["left_elbow_to_hip_vertical"] = keypoints["left_hip"][1] - keypoints["left_elbow"][1]
            current_features["right_elbow_to_hip_vertical"] = keypoints["right_hip"][1] - keypoints["right_elbow"][1]

            # Ankle vs knee (x-axis)
            current_features["left_ankle_to_knee_horizontal"] = abs(keypoints["left_ankle"][0] - keypoints["left_knee"][0])
            current_features["right_ankle_to_knee_horizontal"] = abs(keypoints["right_ankle"][0] - keypoints["right_knee"][0])

            # Torso lean
            current_features["torso_lean"] = calculate_vertical_angle(keypoints["left_shoulder"], keypoints["left_hip"])

            # Stance width / shoulder width
            shoulder_width = calculate_distance(keypoints["left_shoulder"], keypoints["right_shoulder"])
            current_features["stance_width_ratio"] = current_features["feet_distance"] / shoulder_width if shoulder_width > 0 else 0

        except Exception:
            # fallback if something breaks
            for col in feature_names:
                current_features[col] = 0

        # Convert into feature vector with correct order
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
                if label_mapping[predicted_class_index] not in No_Ex_Classes:
                    prediction_label = label_mapping[predicted_class_index]
                else :
                    prediction_label = f"No Exercise"
            else:
                prediction_label = f"No Exercise" 
            
            last_prediction_time = current_time
            
        mp.solutions.drawing_utils.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
    
    else:
        prediction_label = no_person_label
        confidence = 0.0
        frame_sequence = []  
        
    # --- Display Logic ---
    cv2.putText(image, f'Prediction: {prediction_label}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
    cv2.putText(image, f'Confidence: {confidence:.2f}', (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
    
    cv2.imshow(window_name, image)

    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()