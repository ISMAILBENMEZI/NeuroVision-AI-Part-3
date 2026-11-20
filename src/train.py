import time

import torch
from pathlib import Path
from ultralytics import YOLO

from src.config import (
    DATA_YAML,
    MODEL_NAME,
    IMAGE_SIZE,
    BATCH_SIZE,
    EPOCHS,
    PATIENCE,
    RANDOM_STATE,
    RUNS_DIR,
)


def get_device():
    return 0 if torch.cuda.is_available() else "cpu"


def train_model():
    if not DATA_YAML.is_file():
        raise FileNotFoundError(f"Dataset not found: {DATA_YAML}")

    device = get_device()

    model = YOLO(MODEL_NAME)

    start_time = time.time()

    model.train(
        data=str(DATA_YAML),
        epochs=EPOCHS,
        imgsz=IMAGE_SIZE,
        batch=BATCH_SIZE,
        patience=PATIENCE,
        device=device,
        project=str(RUNS_DIR),
        name="yolov8n",
        exist_ok=True,
    )

    training_duration = time.time() - start_time

    best_model_path = RUNS_DIR / "yolov8n" / "weights" / "best.pt"

    if not best_model_path.is_file():
        raise FileNotFoundError("Best model not found.")

    return {
        "best_model_path": best_model_path,
        "training_duration": training_duration,
        "device": device,
    }
