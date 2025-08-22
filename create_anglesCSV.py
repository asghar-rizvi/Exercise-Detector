import pandas as pd
import numpy as np
import math
import os
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing

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

def calculate_vertical_angle(point1, point2):
    """Calculate angle between line formed by two points and vertical axis"""
    dx = point2[0] - point1[0]
    dy = point2[1] - point1[1]
    if dy == 0:
        return 90.0
    angle = math.degrees(math.atan(abs(dx) / abs(dy)))
    return angle

def calculate_symmetry(left_value, right_value):
    """Calculate symmetry between left and right values (0-1 scale)"""
    if left_value == 0 and right_value == 0:
        return 1.0
    return 1.0 - abs(left_value - right_value) / max(left_value, right_value, 1e-6)

POSE_LANDMARKS = {
    "left_shoulder": 11, "right_shoulder": 12,
    "left_elbow": 13, "right_elbow": 14,
    "left_wrist": 15, "right_wrist": 16,
    "left_hip": 23, "right_hip": 24,
    "left_knee": 25, "right_knee": 26,
    "left_ankle": 27, "right_ankle": 28,
    "nose": 0
}

# Include all angles with positive importance
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

def process_row_slim(row_dict):
    features = {}
    
    # Extract all landmark positions
    landmarks = {}
    for name, idx in POSE_LANDMARKS.items():
        landmarks[name] = (row_dict[f"x{idx}"], row_dict[f"y{idx}"])
    
    # Calculate original angles and distances
    for angle_name, (a, b, c) in ANGLE_DEFINITIONS.items():
        features[angle_name] = calculate_angle(landmarks[a], landmarks[b], landmarks[c])
    
    for dist_name, (point1, point2) in DISTANCE_DEFINITIONS.items():
        features[dist_name] = calculate_distance(landmarks[point1], landmarks[point2])
    
    # NEW FEATURES: Only those with positive importance
    
    # Symmetry features (both were important)
    features["elbow_symmetry"] = calculate_symmetry(
        features["left_elbow_angle"], features["right_elbow_angle"]
    )
    features["knee_symmetry"] = calculate_symmetry(
        features["left_knee_angle"], features["right_knee_angle"]
    )
    
    # Hand position relative to shoulders (both were important)
    features["left_hand_to_shoulder_vertical"] = landmarks["left_shoulder"][1] - landmarks["left_wrist"][1]
    features["right_hand_to_shoulder_vertical"] = landmarks["right_shoulder"][1] - landmarks["right_wrist"][1]
    
    # Elbow position relative to hips (both were important)
    features["left_elbow_to_hip_vertical"] = landmarks["left_hip"][1] - landmarks["left_elbow"][1]
    features["right_elbow_to_hip_vertical"] = landmarks["right_hip"][1] - landmarks["right_elbow"][1]
    
    # Ankle position relative to knees (both were important)
    features["left_ankle_to_knee_horizontal"] = abs(landmarks["left_ankle"][0] - landmarks["left_knee"][0])
    features["right_ankle_to_knee_horizontal"] = abs(landmarks["right_ankle"][0] - landmarks["right_knee"][0])
    
    # Torso lean (most important feature)
    features["torso_lean"] = calculate_vertical_angle(
        landmarks["left_shoulder"], landmarks["left_hip"]
    )
    
    # Stance width normalized to shoulder width (had positive importance)
    shoulder_width = calculate_distance(landmarks["left_shoulder"], landmarks["right_shoulder"])
    if shoulder_width > 0:
        features["stance_width_ratio"] = features["feet_distance"] / shoulder_width
    else:
        features["stance_width_ratio"] = 0
    
    features["video_id"] = row_dict["video_id"]
    features["label"] = row_dict["label"]
    return features

def process_csv_revised(input_csv, output_csv):
    print(f"Processing {input_csv} -> {output_csv}")
    df = pd.read_csv(input_csv)
    
    df = df[ (df[f"v{POSE_LANDMARKS['left_hip']}"] > 0.5) & (df[f"v{POSE_LANDMARKS['right_hip']}"] > 0.5) ].copy()
    
    rows = df.to_dict(orient="records")
    
    num_workers = multiprocessing.cpu_count()
    feature_rows = []
    
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = [executor.submit(process_row_slim, row) for row in rows]
        for f in tqdm(as_completed(futures), total=len(futures)):
            feature_rows.append(f.result())
            
    final_df = pd.DataFrame(feature_rows)
    
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    final_df.to_csv(output_csv, index=False)
    print(f"Saved {output_csv}, shape={final_df.shape}")

if __name__ == "__main__":
    process_csv_revised("dataset/keypoints/dataset_train.csv", "dataset/angles/angles_train.csv")
    process_csv_revised("dataset/keypoints/dataset_val.csv", "dataset/angles/angles_val.csv")
    process_csv_revised("dataset/keypoints/dataset_test.csv", "dataset/angles/angles_test.csv")