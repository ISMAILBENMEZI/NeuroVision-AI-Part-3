import mlflow

from mlflow.tracking import MlflowClient
from mlflow.exceptions import MlflowException
from ultralytics import YOLO

from src.config import (
    MODEL_NAME,
    DATA_YAML,
    IMAGE_SIZE,
    BATCH_SIZE,
    EPOCHS,
    PATIENCE,
)

from src.preprocessing import preprocess_dataset
from src.train import train_model


PROJECT_ROOT = DATA_YAML.parents[3]

TRACKING_URI = f"sqlite:///{PROJECT_ROOT / 'mlflow.db'}"

EXPERIMENT_NAME = "NeuroVision-YOLOv8"

REGISTERED_MODEL_NAME = "NeuroVision-Brain-Tumor-Detector"


def preprocessing_task():
    preprocess_dataset()


def training_task():
    mlflow.set_tracking_uri(TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)

    with mlflow.start_run(run_name="yolov8n") as run:
        training = train_model()

        mlflow.log_params(
            {
                "model": MODEL_NAME,
                "epochs": EPOCHS,
                "image_size": IMAGE_SIZE,
                "batch_size": BATCH_SIZE,
                "patience": PATIENCE,
            }
        )

        mlflow.log_metric(
            "training_duration",
            training["training_duration"],
        )

        mlflow.log_artifact(
            str(training["best_model_path"]),
            artifact_path="yolo",
        )

        return {
            "run_id": run.info.run_id,
            "best_model_path": str(training["best_model_path"]),
        }


def evaluation_task(**context):
    training = context["ti"].xcom_pull(task_ids="training")

    mlflow.set_tracking_uri(TRACKING_URI)

    model = YOLO(training["best_model_path"])

    results = model.val(
        data=str(DATA_YAML),
        split="val",
        imgsz=IMAGE_SIZE,
        batch=BATCH_SIZE,
        plots=True,
    )

    metrics = {
        "precision": float(results.box.mp),
        "recall": float(results.box.mr),
        "mAP50": float(results.box.map50),
        "mAP50_95": float(results.box.map),
    }

    client = MlflowClient()

    client.log_metrics(
        run_id=training["run_id"],
        metrics=metrics,
    )

    return metrics


def registration_task(**context):
    training = context["ti"].xcom_pull(task_ids="training")

    metrics = context["ti"].xcom_pull(task_ids="evaluation")

    mlflow.set_tracking_uri(TRACKING_URI)

    client = MlflowClient()

    try:
        client.create_registered_model(REGISTERED_MODEL_NAME)
    except MlflowException:
        pass

    new_version = client.create_model_version(
        name=REGISTERED_MODEL_NAME,
        source=f"runs:/{training['run_id']}/yolo",
        run_id=training["run_id"],
    )

    try:
        champion = client.get_model_version_by_alias(
            REGISTERED_MODEL_NAME,
            "champion",
        )

        old_metrics = client.get_run(champion.run_id).data.metrics

        is_better = (
            metrics["recall"] >= old_metrics["recall"]
            and metrics["mAP50_95"] >= old_metrics["mAP50_95"]
            and (
                metrics["recall"] > old_metrics["recall"]
                or metrics["mAP50_95"] > old_metrics["mAP50_95"]
            )
        )

    except (MlflowException, KeyError):
        is_better = True

    if is_better:
        client.set_registered_model_alias(
            name=REGISTERED_MODEL_NAME,
            alias="champion",
            version=new_version.version,
        )

        print("New champion model registered.")

    else:
        print("Current champion model retained.")

    return {
        "version": new_version.version,
        "champion_updated": is_better,
    }
