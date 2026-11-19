import mlflow
import mlflow.pytorch

import pandas as pd

from pathlib import Path

from src.config import RUNS_DIR


EXPERIMENT_NAME = "NeuroVision-YOLOv8"

REGISTERED_MODEL_NAME = "NeuroVision-Brain-Tumor-Detector"


def setup_mlflow(tracking_uri=None):
    if tracking_uri:
        mlflow.set_tracking_uri(tracking_uri)

    mlflow.set_experiment(EXPERIMENT_NAME)


def log_training_run(
    model,
    params,
    metrics,
    training_duration,
    artifacts=None,
):
    with mlflow.start_run() as run:
        mlflow.log_params(params)

        mlflow.log_metrics(metrics)

        mlflow.log_metric(
            "training_duration",
            training_duration,
        )

        if artifacts:
            for artifact in artifacts:
                artifact_path = Path(artifact)

                if artifact_path.is_file():
                    mlflow.log_artifact(str(artifact_path))

        mlflow.pytorch.log_model(
            model.model,
            name="model",
        )

        return run.info.run_id


def compare_runs():
    experiment = mlflow.get_experiment_by_name(EXPERIMENT_NAME)

    if experiment is None:
        raise ValueError(f"Experiment not found: {EXPERIMENT_NAME}")

    runs = mlflow.search_runs(
        experiment_ids=[experiment.experiment_id],
        order_by=["metrics.recall DESC"],
    )

    if runs.empty:
        print("No experiments found.")
        return runs

    columns = [
        "run_id",
        "params_model",
        "metrics_precision",
        "metrics_recall",
        "metrics_mAP50",
        "metrics_mAP50_95",
        "metrics_training_duration",
    ]

    available_columns = [column for column in columns if column in runs.columns]

    return runs[available_columns]


def register_model(run_id):
    model_uri = f"runs:/{run_id}/model"

    registered_model = mlflow.register_model(
        model_uri=model_uri,
        name=REGISTERED_MODEL_NAME,
    )

    return registered_model


def set_champion_alias(model_version):
    client = mlflow.MlflowClient()

    client.set_registered_model_alias(
        name=REGISTERED_MODEL_NAME,
        alias="champion",
        version=model_version,
    )


def load_champion_model():
    model_uri = f"models:/{REGISTERED_MODEL_NAME}@champion"

    return mlflow.pytorch.load_model(model_uri)
