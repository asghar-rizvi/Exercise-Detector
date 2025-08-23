import os
import shutil
import random
import kagglehub


def getKagglePath():
    path = kagglehub.dataset_download("sharjeelmazhar/human-activity-recognition-video-dataset")
    path2 = os.path.join(path, os.listdir(path)[0])  
    return path2


def create_dirs(dest_dataset, classes, split_counts):
    for split in split_counts.keys():
        for cls in classes:
            os.makedirs(os.path.join(dest_dataset, split, cls), exist_ok=True)


def split_and_copy(src_dataset, dest_dataset, classes, split_counts):
    for cls in classes:
        src_folder = os.path.join(src_dataset, cls)
        if not os.path.exists(src_folder):
            print(f"Source folder missing: {src_folder}")
            continue

        files = [f for f in os.listdir(src_folder) if os.path.isfile(os.path.join(src_folder, f))]
        random.shuffle(files)

        start = 0
        for split, count in split_counts.items():
            split_files = files[start:start+count]
            start += count

            dest_folder = os.path.join(dest_dataset, split, cls)
            for f in split_files:
                shutil.copy(os.path.join(src_folder, f), os.path.join(dest_folder, f))

            print(f"{cls} -> {split}: {len(split_files)} files copied")


if __name__ == "__main__":
    SOURCE_DATASET = getKagglePath()
    DEST_DATASET = "organized_dataset"
    CLASSES = ["Sitting", "Walking", "Standing Still"]
    SPLIT_COUNTS = {
        "train": 60,
        "val": 15,
        "test": 5
    }

    create_dirs(DEST_DATASET, CLASSES, SPLIT_COUNTS)
    split_and_copy(SOURCE_DATASET, DEST_DATASET, CLASSES, SPLIT_COUNTS)

    print("\nDataset successfully organized under:", DEST_DATASET)
