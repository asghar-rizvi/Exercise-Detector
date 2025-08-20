import kagglehub
import os
import shutil

def get_path():
    path = kagglehub.dataset_download("philosopher0808/gym-workoutexercises-video")
    print("Path to dataset files:", path)
    
    return path

def move_exercises(src, dst):
    exercises = [f for f in os.listdir(src) if os.path.isdir(os.path.join(src, f))]
    for ex in exercises:
        src_ex = os.path.join(src, ex)
        dst_ex = os.path.join(dst, ex)
        os.makedirs(dst_ex, exist_ok=True)
        for file in os.listdir(src_ex):
            src_file = os.path.join(src_ex, file)
            dst_file = os.path.join(dst_ex, file)
            shutil.copy2(src_file, dst_file) 


if __name__ == '__main__':
    path = get_path()
    base_path = path
    
    train_src = os.path.join(base_path, "verified_data/verified_data/data_btc_10s")
    val_src = os.path.join(base_path, "verified_data/verified_data/data_crawl_10s")
    test_src = os.path.join(base_path, "test/test")

    target_root = os.path.join(base_path, "organized_dataset")
    train_dst = os.path.join(target_root, "train")
    val_dst = os.path.join(target_root, "val")
    test_dst = os.path.join(target_root, "test")

    for d in [train_dst, val_dst, test_dst]:
        os.makedirs(d, exist_ok=True)
        
    move_exercises(train_src, train_dst)
    move_exercises(val_src, val_dst)
    move_exercises(test_src, test_dst)

    print(" Dataset organized into:", target_root)
    
    print("move into local storage")
    dst = "organized_dataset"
    shutil.copytree(target_root, dst, dirs_exist_ok=True)
    print(f"Dataset copied to: {dst}")        