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

def process_row_slim(row_dict):
    features = {}

    for angle_name, (a, b, c) in ANGLE_DEFINITIONS.items():
        pa = (row_dict[f"x{POSE_LANDMARKS[a]}"], row_dict[f"y{POSE_LANDMARKS[a]}"])
        pb = (row_dict[f"x{POSE_LANDMARKS[b]}"], row_dict[f"y{POSE_LANDMARKS[b]}"])
        pc = (row_dict[f"x{POSE_LANDMARKS[c]}"], row_dict[f"y{POSE_LANDMARKS[c]}"])
        features[angle_name] = calculate_angle(pa, pb, pc)

    for dist_name, (point1, point2) in DISTANCE_DEFINITIONS.items():
        p1 = (row_dict[f"x{POSE_LANDMARKS[point1]}"], row_dict[f"y{POSE_LANDMARKS[point1]}"])
        p2 = (row_dict[f"x{POSE_LANDMARKS[point2]}"], row_dict[f"y{POSE_LANDMARKS[point2]}"])
        features[dist_name] = calculate_distance(p1, p2)
    
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