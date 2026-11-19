import cv2
import mlflow
import numpy as np

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from mlflow.tracking import MlflowClient
from ultralytics import YOLO

MODEL_NAME = "NeuroVision-Brain-Tumor-Detector"
MODEL_ALIAS = "champion"

TRACKING_URI = "sqlite:////home/user/NeuroVision/mlflow.db"

SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}

app = FastAPI(
    title="NeuroVision AI API",
    description="Brain MRI tumor detection using YOLOv8",
    version="1.0.0",
)

model = None


def load_champion_model():
    global model

    mlflow.set_tracking_uri(TRACKING_URI)

    client = MlflowClient()

    champion = client.get_model_version_by_alias(name=MODEL_NAME, alias=MODEL_ALIAS)

    model_path = client.download_artifacts(run_id=champion.run_id, path="yolo/best.pt")

    model = YOLO(model_path)

    print(f"Loaded model version: {champion.version}")
    print(f"Model alias: {MODEL_ALIAS}")


@app.on_event("startup")
def startup_event():
    load_champion_model()


@app.get("/health")
def health():
    return {
        "status": "healthy" if model is not None else "unavailable",
        "model_loaded": model is not None,
        "model_name": MODEL_NAME,
        "model_alias": MODEL_ALIAS,
    }


@app.post("/predict")
async def predict(
    file: UploadFile = File(...), confidence: float = Form(0.25, ge=0.0, le=1.0)
):
    if model is None:
        raise HTTPException(status_code=503, detail="Model is not available.")

    filename = file.filename or ""
    extension = "." + filename.rsplit(".", 1)[-1].lower() if "." in filename else ""

    if extension not in SUPPORTED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail="Unsupported image format. Use JPG, JPEG, PNG or BMP.",
        )

    contents = await file.read()

    if not contents:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")

    image_array = np.frombuffer(contents, dtype=np.uint8)
    image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)

    if image is None:
        raise HTTPException(status_code=422, detail="Invalid or unreadable image.")

    results = model.predict(source=image, conf=confidence, verbose=False)

    detections = []

    for result in results:
        for box in result.boxes:
            class_id = int(box.cls[0])
            class_name = result.names[class_id]
            score = float(box.conf[0])

            x1, y1, x2, y2 = box.xyxy[0].tolist()

            detections.append(
                {
                    "class_id": class_id,
                    "class_name": class_name,
                    "confidence": round(score, 4),
                    "bbox": {
                        "x1": round(x1, 2),
                        "y1": round(y1, 2),
                        "x2": round(x2, 2),
                        "y2": round(y2, 2),
                    },
                }
            )

    return {
        "filename": filename,
        "detections_count": len(detections),
        "message": ("Tumors detected." if detections else "No tumor detected."),
        "detections": detections,
    }
