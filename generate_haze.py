from PIL import Image
import numpy as np
import os
import random


# Original UAVid dataset
input_folder = r"E:\Datasets\UAVID\uavid_train\seq1\Images"

# Project dataset
base_folder = r"E:\DL project\UAVid_Dehazing\dataset"


# Create folders
for split in ["train", "val", "test"]:

    os.makedirs(
        os.path.join(base_folder, split, "hazy"),
        exist_ok=True
    )

    os.makedirs(
        os.path.join(base_folder, split, "GT"),
        exist_ok=True
    )


# Synthetic haze generation
def add_haze(image):

    image = np.array(image).astype(np.float32) / 255.0

    beta = random.uniform(0.5, 1.5)

    transmission = np.exp(-beta)

    atmospheric_light = 0.8

    hazy = (
        image * transmission
        + atmospheric_light * (1 - transmission)
    )

    hazy = np.clip(hazy, 0, 1)

    return (hazy * 255).astype(np.uint8)


# Get images
files = [
    f for f in os.listdir(input_folder)
    if f.lower().endswith(".png")
]


# Shuffle
random.seed(42)
random.shuffle(files)


# 70 / 10 / 20 split
total = len(files)

train_size = int(total * 0.70)
val_size = int(total * 0.10)

train_files = files[:train_size]

val_files = files[
    train_size:train_size + val_size
]

test_files = files[
    train_size + val_size:
]


print("================================")
print("       UAVid DATASET")
print("================================")
print("Total images :", total)
print("Train images :", len(train_files))
print("Val images   :", len(val_files))
print("Test images  :", len(test_files))
print("================================")


# Generate datasets
for split, split_files in [
    ("train", train_files),
    ("val", val_files),
    ("test", test_files)
]:

    print("\nCreating", split, "data...")

    for i, file in enumerate(split_files):

        path = os.path.join(input_folder, file)

        image = Image.open(path).convert("RGB")

        # Original image = ground truth
        image.save(
            os.path.join(
                base_folder,
                split,
                "GT",
                file
            )
        )

        # Generate synthetic haze
        hazy = add_haze(image)

        Image.fromarray(hazy).save(
            os.path.join(
                base_folder,
                split,
                "hazy",
                file
            )
        )

        if (i + 1) % 50 == 0:
            print(
                split,
                ":",
                i + 1,
                "/",
                len(split_files)
            )


print("\n================================")
print("Dataset creation completed!")
print("================================")