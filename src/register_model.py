import mlflow

from mlflow.exceptions import MlflowException
from mlflow.tracking import MlflowClient

from src.config import PROJECT_ROOT

RUN_ID = "98cc7b2839e1472880bd0fd1dc975294"

MODEL_PATH = (
    PROJECT_ROOT / "runs" / "detect" / "runs" / "quick_test" / "weights" / "best.pt"
)

MODEL_NAME = "NeuroVision-Brain-Tumor-Detector"

mlflow.set_tracking_uri("sqlite:////home/user/NeuroVision/mlflow.db")

client = MlflowClient()

if not MODEL_PATH.exists():
    raise FileNotFoundError(f"Model not found: {MODEL_PATH}")

try:
    client.create_registered_model(MODEL_NAME)
    print("Registered Model created.")

except MlflowException as error:
    if "already exists" in str(error).lower():
        print("Registered Model already exists.")
    else:
        raise

client.log_artifact(run_id=RUN_ID, local_path=str(MODEL_PATH), artifact_path="yolo")

model_version = client.create_model_version(
    name=MODEL_NAME, source=f"runs:/{RUN_ID}/yolo", run_id=RUN_ID
)

client.set_registered_model_alias(
    name=MODEL_NAME, alias="champion", version=model_version.version
)

champion = client.get_model_version_by_alias(name=MODEL_NAME, alias="champion")

print("\nModel registered successfully!")
print("Model:", MODEL_NAME)
print("Version:", champion.version)
print("Alias: champion")
print("Artifact: yolo/best.pt")
