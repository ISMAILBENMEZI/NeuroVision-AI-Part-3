
#!/bin/bash

cd /home/user/NeuroVision

export AIRFLOW_HOME=/home/user/NeuroVision/airflow
export AIRFLOW__CORE__DAGS_FOLDER=/home/user/NeuroVision/dags

mkdir -p "$AIRFLOW_HOME/logs"

AIRFLOW=".venv/bin/airflow"

echo "Starting Airflow Scheduler..."

"$AIRFLOW" scheduler --daemon

echo "Starting Airflow Webserver..."

"$AIRFLOW" webserver --port 8085 --daemon

echo "Airflow started successfully!"
echo "Open: http://localhost:8085"
