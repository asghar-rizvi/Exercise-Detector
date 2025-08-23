import os
os.environ["GLOG_minloglevel"] = "2"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import cv2
import mediapipe as mp
import pandas as pd
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed  # use processes

mp_pose = mp.solutions.pose

def extract_keypoints_from_video(video_path, frame_skip=5):
    cap = cv2.VideoCapture(video_path)
    pose = mp_pose.Pose(
        static_image_mode=False, 
        min_detection_confidence=0.3, 
        min_tracking_confidence=0.3
    )

    keypoints_data = []
    frame_count = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        if frame_count % frame_skip != 0:  
            continue
        frame = cv2.resize(frame, (640, 360))
        
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(frame_rgb)

        if results.pose_landmarks:
            landmarks = results.pose_landmarks.landmark
            keypoints = []
            for lm in landmarks:
                keypoints.extend([lm.x, lm.y, lm.z, lm.visibility])
            keypoints_data.append(keypoints)
        else:
            keypoints_data.append([0] * (33*4))

    cap.release()
    pose.close()   
    del pose
    return keypoints_data

def process_one_video(args):
    video_path, exercise = args
    video_id = os.path.basename(video_path)
    keypoints_data = extract_keypoints_from_video(video_path)
    rows = []
    for frame_keypoints in keypoints_data:
        row = frame_keypoints + [exercise, video_id]
        rows.append(row)
    return rows

def process_dataset_parallel(dataset_path, output_csv):
    data_rows = []
    tasks = []

    for exercise in os.listdir(dataset_path):
        exercise_path = os.path.join(dataset_path, exercise)
        if not os.path.isdir(exercise_path):
            continue

        for video in os.listdir(exercise_path):
            video_path = os.path.join(exercise_path, video)
            if video_path.lower().endswith(('.mp4', '.avi', '.mov', '.mkv')):
                tasks.append((video_path, exercise))

    num_workers = 6
    print(f"Using {num_workers} processes in parallel...")

    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = [executor.submit(process_one_video, task) for task in tasks]

        for f in tqdm(as_completed(futures), total=len(futures)):
            data_rows.extend(f.result())

    columns = []
    for i in range(33):
        columns.extend([f"x{i}", f"y{i}", f"z{i}", f"v{i}"])
    columns.append("label")
    columns.append("video_id")

    df = pd.DataFrame(data_rows, columns=columns)
    df.to_csv(output_csv, index=False)
    print(f"Saved {output_csv}, shape: {df.shape}")

if __name__ == '__main__':
    base_path = "organized_dataset"
    
    process_dataset_parallel(os.path.join(base_path, "train"), "dataset/keypoints/dataset_train.csv")
    process_dataset_parallel(os.path.join(base_path, "val"), "dataset/keypoints/dataset_val.csv")
    process_dataset_parallel(os.path.join(base_path, "test"), "dataset/keypoints/dataset_test.csv")
    
    print('Dataset completed')
