import sys

from pathlib import Path
from datetime import datetime, timedelta

from airflow import DAG
from airflow.operators.python import PythonOperator


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.airflow_tasks import (
    preprocessing_task,
    training_task,
    evaluation_task,
    registration_task,
)


default_args = {
    "owner": "neurovision",
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
}


with DAG(
    dag_id="neurovision_pipeline",
    description="NeuroVision AI YOLOv8 pipeline",
    default_args=default_args,
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
