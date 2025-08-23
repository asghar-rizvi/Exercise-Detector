import os
import shutil
import random

base_path = "organized_dataset"

target_val_count = 15
try:
    val_base_path = os.path.join(base_path, "val")
    exercise_folders = [d for d in os.listdir(val_base_path) if os.path.isdir(os.path.join(val_base_path, d))]
except FileNotFoundError:
    print(f"Error: The 'val' directory was not found at {val_base_path}")
    exit()

for exercise_folder in exercise_folders:
    print(f"--- Processing '{exercise_folder}' folder ---")

    val_path = os.path.join(base_path, "val", exercise_folder)
    train_path = os.path.join(base_path, "train", exercise_folder)

    if not os.path.exists(train_path):
        print(f"Warning: Corresponding train path does not exist for '{exercise_folder}'. Skipping.")
        continue

    all_val_videos = [f for f in os.listdir(val_path) if os.path.isfile(os.path.join(val_path, f))]
    current_val_count = len(all_val_videos)

    print(f"Found {current_val_count} videos in the validation folder for '{exercise_folder}'.")

    if current_val_count <= target_val_count:
        print(f"The number of videos ({current_val_count}) is already {target_val_count} or less. No videos will be moved.")
    else:
        num_to_move = current_val_count - target_val_count
        print(f"Need to move {num_to_move} videos to the training folder.")

        videos_to_move = random.sample(all_val_videos, num_to_move)

        for video_filename in videos_to_move:
            source_file = os.path.join(val_path, video_filename)
            destination_file = os.path.join(train_path, video_filename)
            
            try:
                shutil.move(source_file, destination_file)
            except Exception as e:
                print(f"Error moving {video_filename}: {e}")

        # Verify the final count.
        remaining_val_videos = [f for f in os.listdir(val_path) if os.path.isfile(os.path.join(val_path, f))]
        print(f"Final count in validation folder: {len(remaining_val_videos)} videos.")
    print("-" * 30)

print("\nProcess completed for all exercise folders!")
