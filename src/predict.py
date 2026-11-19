from pathlib import Path

from ultralytics import YOLO

from src.config import (
    RUNS_DIR,
    IMAGE_SIZE,
    CONFIDENCE_THRESHOLD,
    SUPPORTED_EXTENSIONS,
)


BEST_MODEL_PATH = RUNS_DIR / "yolov8n_baseline" / "weights" / "best.pt"

PREDICTIONS_DIR = RUNS_DIR / "predictions"


def load_model(model_path=BEST_MODEL_PATH):
    model_path = Path(model_path)

    if not model_path.is_file():
        raise FileNotFoundError(f"Model weights not found: {model_path}")

    return YOLO(str(model_path))


def validate_image(image_path):
    image_path = Path(image_path)

    if not image_path.is_file():
        raise FileNotFoundError(f"Image not found: {image_path}")

    if image_path.suffix.lower() not in SUPPORTED_EXTENSIONS:
        raise ValueError(f"Unsupported image format: {image_path.suffix}")

    return image_path


def predict_image(
    model,
    image_path,
    confidence=CONFIDENCE_THRESHOLD,
):
    image_path = validate_image(image_path)

    results = model.predict(
        source=str(image_path),
        imgsz=IMAGE_SIZE,
        conf=confidence,
        save=False,
        verbose=False,
    )

    predictions = []

    for result in results:
        for box in result.boxes:
            class_id = int(box.cls.item())

            predictions.append(
                {
                    "class_id": class_id,
                    "class_name": model.names[class_id],
                    "confidence": float(box.conf.item()),
                    "bbox": {
                        "x1": float(box.xyxy[0][0].item()),
                        "y1": float(box.xyxy[0][1].item()),
                        "x2": float(box.xyxy[0][2].item()),
                        "y2": float(box.xyxy[0][3].item()),
                    },
                }
            )

    return {
        "image": image_path.name,
        "detections": predictions,
        "detection_count": len(predictions),
        "message": ("Tumor detected" if predictions else "No tumor detected"),
    }


def save_prediction(
    model,
    image_path,
    confidence=CONFIDENCE_THRESHOLD,
):
    image_path = validate_image(image_path)

    PREDICTIONS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    results = model.predict(
        source=str(image_path),
        imgsz=IMAGE_SIZE,
        conf=confidence,
        save=True,
        project=str(PREDICTIONS_DIR),
        name="results",
        exist_ok=True,
        verbose=False,
    )

    return Path(results[0].save_dir)
