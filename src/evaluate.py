from pathlib import Path

import pandas as pd
from ultralytics import YOLO

from src.config import (
    DATA_YAML,
    RUNS_DIR,
    IMAGE_SIZE,
    BATCH_SIZE,
    CONFIDENCE_THRESHOLD,
)


BEST_MODEL_PATH = RUNS_DIR / "yolov8n_baseline" / "weights" / "best.pt"

EVALUATION_DIR = RUNS_DIR / "evaluation"


def load_model(model_path=BEST_MODEL_PATH):
    if not model_path.is_file():
        raise FileNotFoundError(f"Model weights not found: {model_path}")

    return YOLO(str(model_path))


def evaluate_model(model):
    if not DATA_YAML.is_file():
        raise FileNotFoundError(f"Dataset configuration not found: {DATA_YAML}")

    results = model.val(
        data=str(DATA_YAML),
        split="test",
        imgsz=IMAGE_SIZE,
        batch=BATCH_SIZE,
        conf=CONFIDENCE_THRESHOLD,
        plots=True,
        save_json=True,
        project=str(RUNS_DIR),
        name="evaluation",
        exist_ok=True,
    )

    return results


def extract_metrics(results):
    metrics = {
        "precision": float(results.box.mp),
        "recall": float(results.box.mr),
        "mAP50": float(results.box.map50),
        "mAP50_95": float(results.box.map),
    }

    class_metrics = {}

    for class_id, class_name in results.names.items():
        class_metrics[class_name] = {
            "precision": float(results.box.p[class_id]),
            "recall": float(results.box.r[class_id]),
            "mAP50": float(results.box.ap50[class_id]),
            "mAP50_95": float(results.box.ap[class_id]),
        }

    return metrics, class_metrics


def save_metrics(metrics, class_metrics):
    EVALUATION_DIR.mkdir(parents=True, exist_ok=True)

    pd.DataFrame([metrics]).to_csv(
        EVALUATION_DIR / "overall_metrics.csv",
        index=False,
    )

    class_df = pd.DataFrame.from_dict(
        class_metrics,
        orient="index",
    )

    class_df.index.name = "class"

    class_df.to_csv(EVALUATION_DIR / "class_metrics.csv")


def print_metrics(metrics, class_metrics):
    print("\nOverall Evaluation Metrics")
    print("-" * 40)

    for metric_name, value in metrics.items():
        print(f"{metric_name}: {value:.4f}")

    print("\nClass-wise Evaluation Metrics")
    print("-" * 40)

    for class_name, values in class_metrics.items():
        print(f"\n{class_name}")

        for metric_name, value in values.items():
            print(f"  {metric_name}: {value:.4f}")


def evaluate():
    model = load_model()

    results = evaluate_model(model)

    metrics, class_metrics = extract_metrics(results)

    save_metrics(metrics, class_metrics)

    print_metrics(metrics, class_metrics)

    return {
        "results": results,
        "metrics": metrics,
        "class_metrics": class_metrics,
    }
