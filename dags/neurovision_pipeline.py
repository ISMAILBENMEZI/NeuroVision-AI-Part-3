import sys
from pathlib import Path
from datetime import datetime, timedelta

from airflow import DAG
from airflow.operators.python import PythonOperator


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

TRACKING_URI = f"sqlite:///{PROJECT_ROOT / 'mlflow.db'}"
EXPERIMENT_NAME = "NeuroVision-YOLOv8"
REGISTERED_MODEL_NAME = "NeuroVision-Brain-Tumor-Detector"


def preprocessing_task():
    from src.preprocessing import preprocess_dataset

    preprocess_dataset()


def training_task():
    import mlflow
    from src.train import train_model
    from src.config import (
        MODEL_NAME,
        IMAGE_SIZE,
        BATCH_SIZE,
        EPOCHS,
        PATIENCE,
        RANDOM_STATE,
        MOSAIC,
    )

    mlflow.set_tracking_uri(TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)

    training = train_model()

    best_model_path = Path(training["best_model_path"])

    with mlflow.start_run(run_name="yolov8n_airflow") as run:
        mlflow.log_params(
            {
                "model": MODEL_NAME,
                "image_size": IMAGE_SIZE,
                "batch_size": BATCH_SIZE,
                "epochs": EPOCHS,
                "patience": PATIENCE,
                "seed": RANDOM_STATE,
                "mosaic": MOSAIC,
                "device": str(training["device"]),
            }
        )

        mlflow.log_metric(
            "training_duration",
            training["training_duration"],
        )

        mlflow.log_artifact(
            str(best_model_path),
            artifact_path="yolo",
        )

        return {
            "run_id": run.info.run_id,
            "best_model_path": str(best_model_path),
        }


def evaluation_task(**context):
    import mlflow
    from mlflow.tracking import MlflowClient
    from ultralytics import YOLO
    from src.config import (
        DATA_YAML,
        IMAGE_SIZE,
        BATCH_SIZE,
        CONFIDENCE_THRESHOLD,
        RUNS_DIR,
    )

    training = context["ti"].xcom_pull(task_ids="training")

    mlflow.set_tracking_uri(TRACKING_URI)

    model = YOLO(training["best_model_path"])

    results = model.val(
        data=str(DATA_YAML),
        split="val",
        imgsz=IMAGE_SIZE,
        batch=BATCH_SIZE,
        conf=CONFIDENCE_THRESHOLD,
        plots=True,
        project=str(RUNS_DIR),
        name="airflow_evaluation",
        exist_ok=True,
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
    import mlflow
    from mlflow.exceptions import MlflowException
    from mlflow.tracking import MlflowClient

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

    update_champion = True

    try:
        champion = client.get_model_version_by_alias(
            REGISTERED_MODEL_NAME,
            "champion",
        )

        champion_metrics = client.get_run(champion.run_id).data.metrics

        old_recall = champion_metrics.get("recall")
        old_map = champion_metrics.get("mAP50_95")

        if old_recall is not None and old_map is not None:
            update_champion = (
                metrics["recall"] >= old_recall
                and metrics["mAP50_95"] >= old_map
                and (metrics["recall"] > old_recall or metrics["mAP50_95"] > old_map)
            )

    except MlflowException:
        update_champion = True

    if update_champion:
        client.set_registered_model_alias(
            name=REGISTERED_MODEL_NAME,
            alias="champion",
            version=new_version.version,
        )
        print(f"Champion updated to version {new_version.version}")
    else:
        print("Current champion is retained.")

    return {
        "version": new_version.version,
        "champion_updated": update_champion,
    }


default_args = {
    "owner": "neurovision",
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
}

with DAG(
    dag_id="neurovision_pipeline",
    default_args=default_args,
    description="NeuroVision AI YOLOv8 pipeline",
    start_date=datetime(2026, 11, 19),
    schedule=None,
    catchup=False,
    max_active_runs=1,
    tags=["NeuroVision", "YOLOv8", "MLflow"],
) as dag:
    preprocessing = PythonOperator(
        task_id="preprocessing",
        python_callable=preprocessing_task,
    )

    training = PythonOperator(
        task_id="training",
        python_callable=training_task,
    )

    evaluation = PythonOperator(
        task_id="evaluation",
        python_callable=evaluation_task,
    )

    registration = PythonOperator(
        task_id="registration",
        python_callable=registration_task,
    )

    preprocessing >> training >> evaluation >> registration
