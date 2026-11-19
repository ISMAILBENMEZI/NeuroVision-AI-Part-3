import shutil
from pathlib import Path

import cv2
import kagglehub
import numpy as np
import yaml

from sklearn.model_selection import train_test_split

from src.config import (
    DATASET_ID,
    DATASET_DIR,
    DATA_YAML,
    CLASS_NAMES,
    NUM_CLASSES,
    SUPPORTED_EXTENSIONS,
    RANDOM_STATE,
    TRAIN_SIZE,
    TEST_SIZE,
)


def load_dataset():
    dataset_path = Path(kagglehub.dataset_download(DATASET_ID))

    if not dataset_path.is_dir():
        raise FileNotFoundError(f"Dataset directory not found: {dataset_path}")

    return dataset_path


def find_source_splits(dataset_path):
    split_directories = {}

    for split_name in ["Train", "Val"]:
        matching_paths = [
            path
            for path in dataset_path.rglob("*")
            if path.is_dir() and path.name.lower() == split_name.lower()
        ]

        if not matching_paths:
            raise FileNotFoundError(f"{split_name} directory not found.")

        split_directories[split_name.lower()] = matching_paths[0]

    return split_directories


def validate_image(image_path):
    image = cv2.imread(str(image_path))

    return image is not None


def validate_annotation(label_path):
    if not label_path.is_file():
        return False

    try:
        with open(label_path, "r", encoding="utf-8") as file:
            lines = file.readlines()

        for line in lines:
            values = line.strip().split()

            if len(values) != 5:
                return False

            class_id = int(values[0])

            if not 0 <= class_id < NUM_CLASSES:
                return False

            coordinates = np.array([float(value) for value in values[1:]], dtype=float)

            if not np.all(np.isfinite(coordinates)):
                return False

            if not np.all((coordinates >= 0) & (coordinates <= 1)):
                return False

            _, _, width, height = coordinates

            if width <= 0 or height <= 0:
                return False

        return True

    except (ValueError, OSError):
        return False


def collect_samples(split_directory):
    samples = []

    for class_name in CLASS_NAMES:
        class_directories = [
            path
            for path in split_directory.iterdir()
            if path.is_dir() and path.name.lower() == class_name.lower()
        ]

        if not class_directories:
            raise FileNotFoundError(f"Class directory not found: {class_name}")

        class_directory = class_directories[0]

        images_directory = class_directory / "images"
        labels_directory = class_directory / "labels"

        if not images_directory.is_dir():
            raise FileNotFoundError(f"Images directory not found: {images_directory}")

        if not labels_directory.is_dir():
            raise FileNotFoundError(f"Labels directory not found: {labels_directory}")

        for image_path in images_directory.rglob("*"):
            if not image_path.is_file():
                continue

            if image_path.suffix.lower() not in SUPPORTED_EXTENSIONS:
                continue

            label_path = labels_directory / f"{image_path.stem}.txt"

            if not validate_image(image_path):
                continue

            if not validate_annotation(label_path):
                continue

            samples.append(
                {
                    "image": image_path,
                    "label": label_path,
                    "class_name": class_name,
                }
            )

    if not samples:
        raise ValueError(f"No valid samples found in {split_directory}")

    return samples


def split_dataset(train_samples):
    labels = [sample["class_name"] for sample in train_samples]

    train_data, test_data = train_test_split(
        train_samples,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=labels,
    )

    return train_data, test_data


def organize_dataset(dataset):
    for split_name, samples in dataset.items():
        images_directory = DATASET_DIR / "images" / split_name
        labels_directory = DATASET_DIR / "labels" / split_name

        images_directory.mkdir(parents=True, exist_ok=True)
        labels_directory.mkdir(parents=True, exist_ok=True)

        for index, sample in enumerate(samples):
            image_path = sample["image"]
            label_path = sample["label"]
            class_name = sample["class_name"]

            new_name = f"{class_name}_{image_path.stem}_{index}"

            destination_image = (
                images_directory / f"{new_name}{image_path.suffix.lower()}"
            )

            destination_label = labels_directory / f"{new_name}.txt"

            shutil.copy2(image_path, destination_image)
            shutil.copy2(label_path, destination_label)


# def organize_dataset(dataset):
#     if DATASET_DIR.exists():
#         shutil.rmtree(DATASET_DIR)

#     for split_name, samples in dataset.items():
#         images_directory = DATASET_DIR / "images" / split_name
#         labels_directory = DATASET_DIR / "labels" / split_name

#         images_directory.mkdir(parents=True, exist_ok=True)
#         labels_directory.mkdir(parents=True, exist_ok=True)

#         for index, sample in enumerate(samples):
#             image_path = sample["image"]
#             label_path = sample["label"]

#             new_name = f"{sample['class_name']}_{image_path.stem}_{index}"

#             shutil.copy2(
#                 image_path, images_directory / f"{new_name}{image_path.suffix.lower()}"
#             )

#             shutil.copy2(label_path, labels_directory / f"{new_name}.txt")


def create_data_yaml():
    data = {
        "path": str(DATASET_DIR.resolve()),
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",
        "nc": NUM_CLASSES,
        "names": {index: name for index, name in enumerate(CLASS_NAMES)},
    }

    DATASET_DIR.mkdir(parents=True, exist_ok=True)

    with open(DATA_YAML, "w", encoding="utf-8") as file:
        yaml.safe_dump(
            data,
            file,
            sort_keys=False,
            allow_unicode=True,
        )


def preprocess_dataset():
    dataset_path = load_dataset()

    source_splits = find_source_splits(dataset_path)

    train_samples = collect_samples(source_splits["train"])

    val_samples = collect_samples(source_splits["val"])

    train_data, test_data = split_dataset(train_samples)

    dataset = {
        "train": train_data,
        "val": val_samples,
        "test": test_data,
    }

    organize_dataset(dataset)

    create_data_yaml()

    return dataset
