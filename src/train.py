import time

import torch
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
    DEGREES,
    TRANSLATE,
    SCALE,
    FLIPUD,
    FLIPLR,
    HSV_H,
    HSV_S,
    HSV_V,
    MOSAIC,
)


def get_device():
    return 0 if torch.cuda.is_available() else "cpu"


def train_model():
    if not DATA_YAML.is_file():
        raise FileNotFoundError(f"Dataset configuration not found: {DATA_YAML}")

    device = get_device()

    model = YOLO(MODEL_NAME)

    start_time = time.time()

    results = model.train(
        data=str(DATA_YAML),
        epochs=EPOCHS,
        imgsz=IMAGE_SIZE,
        batch=BATCH_SIZE,
        patience=PATIENCE,
        seed=RANDOM_STATE,
        device=device,
        project=str(RUNS_DIR),
        name="yolov8n_baseline",
        exist_ok=True,
        plots=True,
        degrees=DEGREES,
        translate=TRANSLATE,
        scale=SCALE,
        flipud=FLIPUD,
        fliplr=FLIPLR,
        hsv_h=HSV_H,
        hsv_s=HSV_S,
        hsv_v=HSV_V,
        mosaic=MOSAIC,
    )

    training_duration = time.time() - start_time

    best_model_path = RUNS_DIR / "yolov8n_baseline" / "weights" / "best.pt"

    last_model_path = RUNS_DIR / "yolov8n_baseline" / "weights" / "last.pt"

    if not best_model_path.is_file():
        raise FileNotFoundError(f"Best model not found: {best_model_path}")

    return {
        "results": results,
        "best_model_path": best_model_path,
        "last_model_path": last_model_path,
        "training_duration": training_duration,
        "device": device,
    }
