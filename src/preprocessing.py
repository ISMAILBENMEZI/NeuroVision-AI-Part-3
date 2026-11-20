import os
import shutil
import tempfile
from pathlib import Path

import cv2
import kagglehub
import numpy as np
import yaml

from sklearn.model_selection import train_test_split

from src.config import (
    DATASET_ID,
    DATASET_DIR,
    CLASS_NAMES,
    NUM_CLASSES,
    SUPPORTED_EXTENSIONS,
    RANDOM_STATE,
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

        if len(matching_paths) > 1:
            raise ValueError(
                f"Multiple {split_name} directories found: {matching_paths}"
            )

        split_directories[split_name.lower()] = matching_paths[0]

    return split_directories


def validate_image(image_path):
    try:
        image = cv2.imread(str(image_path))
        return image is not None
    except cv2.error:
        return False


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

            coordinates = np.array(
                [float(value) for value in values[1:]],
                dtype=float,
            )

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

        if len(class_directories) != 1:
            raise ValueError(
                f"Expected one directory for class {class_name}, "
                f"found {len(class_directories)}."
            )

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


def organize_dataset(dataset, destination_dir):
    for split_name, samples in dataset.items():
        images_directory = destination_dir / "images" / split_name

        labels_directory = destination_dir / "labels" / split_name

        images_directory.mkdir(
            parents=True,
            exist_ok=True,
        )

        labels_directory.mkdir(
            parents=True,
            exist_ok=True,
        )

        for index, sample in enumerate(samples):
            image_path = sample["image"]
            label_path = sample["label"]
            class_name = sample["class_name"]

            new_name = f"{class_name}_{image_path.stem}_{index}"

            destination_image = (
                images_directory / f"{new_name}{image_path.suffix.lower()}"
            )

            destination_label = labels_directory / f"{new_name}.txt"

            shutil.copy2(
                image_path,
                destination_image,
            )

            shutil.copy2(
                label_path,
                destination_label,
            )


def create_data_yaml(dataset_dir):
    data = {
        "path": str(DATASET_DIR.resolve()),
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",
        "nc": NUM_CLASSES,
        "names": {index: name for index, name in enumerate(CLASS_NAMES)},
    }

    yaml_path = dataset_dir / "data.yaml"

    with open(yaml_path, "w", encoding="utf-8") as file:
        yaml.safe_dump(
            data,
            file,
            sort_keys=False,
            allow_unicode=True,
        )


def validate_processed_dataset(dataset_dir):
    for split_name in ["train", "val", "test"]:
        images_directory = dataset_dir / "images" / split_name

        labels_directory = dataset_dir / "labels" / split_name

        if not images_directory.is_dir():
            raise FileNotFoundError(f"Missing images directory: {images_directory}")

        if not labels_directory.is_dir():
            raise FileNotFoundError(f"Missing labels directory: {labels_directory}")

        images = [path for path in images_directory.iterdir() if path.is_file()]

        labels = [path for path in labels_directory.iterdir() if path.is_file()]

        if len(images) != len(labels):
            raise ValueError(
                f"Image-label mismatch in {split_name}: "
                f"{len(images)} images, {len(labels)} labels."
            )

        image_stems = {path.stem for path in images}
        label_stems = {path.stem for path in labels}

        if image_stems != label_stems:
            raise ValueError(f"Image-label filenames do not match in {split_name}.")

        print(f"{split_name}: {len(images)} images and {len(labels)} labels")

    yaml_path = dataset_dir / "data.yaml"

    if not yaml_path.is_file():
        raise FileNotFoundError(f"Dataset configuration not found: {yaml_path}")

    with open(yaml_path, "r", encoding="utf-8") as file:
        data = yaml.safe_load(file)

    if data["nc"] != NUM_CLASSES:
        raise ValueError("Invalid number of classes in data.yaml.")

    if len(data["names"]) != NUM_CLASSES:
        raise ValueError("Invalid class names in data.yaml.")


def replace_dataset(staging_dir):
    backup_dir = DATASET_DIR.with_name(f"{DATASET_DIR.name}.backup")

    if backup_dir.exists():
        raise FileExistsError(
            f"Backup already exists: {backup_dir}. "
            "Preserve it and resolve it before replacing the dataset."
        )

    old_dataset_exists = DATASET_DIR.exists()

    try:
        if old_dataset_exists:
            os.replace(DATASET_DIR, backup_dir)

        os.replace(staging_dir, DATASET_DIR)

    except Exception:
        if not DATASET_DIR.exists() and backup_dir.exists():
            os.replace(backup_dir, DATASET_DIR)

        raise


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

    DATASET_DIR.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    staging_dir = Path(
        tempfile.mkdtemp(
            prefix="brain_mri_staging_",
            dir=DATASET_DIR.parent,
        )
    )

    try:
        organize_dataset(
            dataset,
            staging_dir,
        )

        create_data_yaml(staging_dir)

        validate_processed_dataset(staging_dir)

        replace_dataset(staging_dir)

    finally:
        if staging_dir.exists():
            shutil.rmtree(staging_dir)

    return {split_name: len(samples) for split_name, samples in dataset.items()}
