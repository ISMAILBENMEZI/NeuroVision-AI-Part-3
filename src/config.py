
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

DATASET_DIR = PROCESSED_DATA_DIR / "brain_mri_yolo"
DATA_YAML = DATASET_DIR / "data.yaml"

MODELS_DIR = PROJECT_ROOT / "models"
RUNS_DIR = PROJECT_ROOT / "runs"



DATASET_ID = "ahmedsorour1/mri-for-brain-tumor-with-bounding-boxes"


CLASS_NAMES = [
    "Glioma",
    "Meningioma",
    "No Tumor",
    "Pituitary"
]

NUM_CLASSES = len(CLASS_NAMES)

SUPPORTED_EXTENSIONS = (
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp"
)



RANDOM_STATE = 42

TRAIN_SIZE = 0.80
TEST_SIZE = 0.20


MODEL_NAME = "yolov8n.pt"

IMAGE_SIZE = 224
BATCH_SIZE = 16

EPOCHS = 50
PATIENCE = 10

CONFIDENCE_THRESHOLD = 0.25


DEGREES = 5.0
TRANSLATE = 0.05
SCALE = 0.10

FLIPUD = 0.0
FLIPLR = 0.0

HSV_H = 0.0
HSV_S = 0.0
HSV_V = 0.10

MOSAIC = 0.0
