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
    gru_model = load_model('files/GRU_model_3.h5')
    scaler = joblib.load('files/real_time_scaler.pkl')
    with open('files/encoded_labels.json', 'r') as f:
        label_mapping = json.load(f)
        label_mapping = {int(k): v for k, v in label_mapping.items()}
    print("GRU model and preprocessing files loaded successfully!")
except FileNotFoundError as e:
    print(f"Error: {e}. Make sure you have downloaded the required files (gru_model.h5, real_time_scaler.pkl, encoded_labels.json) and placed them in the same directory.")
    exit()
except Exception as e:
    print(f"An unexpected error occurred: {e}")
    exit()

# --- Step 2: Define Landmarks and the Exact Features Used for Training ---
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)

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
    "left_hip_angle": ("left_shoulder", "left_hip", "left_knee"),
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

def calculate_angle(a, b, c):
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)
    ba = a - b
    bc = c - b
    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-6)
    angle = np.arccos(np.clip(cosine_angle, -1.0, 1.0))
    return np.degrees(angle)

def calculate_distance(point1, point2):
    return math.sqrt((point2[0] - point1[0])**2 + (point2[1] - point1[1])**2)

sequence_length = 25
frame_sequence = []
prediction_label = "No Detection"
confidence = 0.0
# Define the label for "No person detected"
no_person_label = "No Person Detected"

feature_names = list(ANGLE_DEFINITIONS.keys()) + list(DISTANCE_DEFINITIONS.keys())

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
        # A person is detected, so proceed with feature extraction and prediction.
        current_features = {}
        try:
            landmarks = results.pose_landmarks.landmark
            keypoints = {
                name: [landmarks[id].x, landmarks[id].y] for name, id in POSE_LANDMARKS.items()
            }

            for angle_name, (a, b, c) in ANGLE_DEFINITIONS.items():
                pa = keypoints[a]
                pb = keypoints[b]
                pc = keypoints[c]
                current_features[angle_name] = calculate_angle(pa, pb, pc)
            
            for dist_name, (p1, p2) in DISTANCE_DEFINITIONS.items():
                current_features[dist_name] = calculate_distance(keypoints[p1], keypoints[p2])

        except Exception:
            # Fallback for unexpected errors during feature extraction
            for col in feature_names:
                current_features[col] = 0

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
            
            prediction_label = label_mapping[predicted_class_index]
            confidence = prediction_probabilities[predicted_class_index]
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