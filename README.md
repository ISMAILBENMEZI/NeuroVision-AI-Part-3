# NeuroVision AI 🧠
### Brain MRI Tumor Detection with YOLOv8, MLflow, FastAPI, Streamlit & Apache Airflow

![Python](https://img.shields.io/badge/Python-3.12-blue?logo=python)
![YOLOv8](https://img.shields.io/badge/YOLOv8-Ultralytics-purple)
![OpenCV](https://img.shields.io/badge/OpenCV-Image%20Processing-green?logo=opencv)
![MLflow](https://img.shields.io/badge/MLflow-Experiment%20Tracking-blue?logo=mlflow)
![Airflow](https://img.shields.io/badge/Apache%20Airflow-Pipeline%20Orchestration-red?logo=apacheairflow)
![FastAPI](https://img.shields.io/badge/FastAPI-API-009688?logo=fastapi)
![Streamlit](https://img.shields.io/badge/Streamlit-Interface-FF4B4B?logo=streamlit)

---

## 1. Project Overview

**NeuroVision AI** is a computer vision project designed to assist in the analysis of brain MRI images by detecting and localizing potential brain tumors.

The project is developed in three parts, progressively moving from image classification to object detection and deployment.

- **Part 1 – CNN from Scratch:** Building a Convolutional Neural Network (CNN) to classify brain MRI images into four categories.
- **Part 2 – Transfer Learning:** Using MobileNetV2 with Transfer Learning and Fine-Tuning to improve image classification performance.
- **Part 3 – Object Detection:** Using YOLOv8 to detect and localize potential tumors in MRI images with bounding boxes, predicted classes, and confidence scores.

Unlike image classification, which assigns a label to an entire image, object detection identifies *where* an object is located and *what* it represents.

The third part also introduces an end-to-end Machine Learning workflow:

- Dataset preparation and validation.
- YOLOv8 model training.
- Model evaluation and error analysis.
- Experiment tracking with MLflow.
- Model versioning and registration.
- Model serving through FastAPI.
- Interactive testing through Streamlit.
- Pipeline automation with Apache Airflow.

The objective is to build a reproducible, documented, and maintainable computer vision workflow.

> **Medical Disclaimer:** NeuroVision AI is an educational and experimental project. It is intended to assist research and medical image analysis, not to provide a medical diagnosis or replace qualified healthcare professionals.

---

## 2. Project Objectives

The main objectives of this project are:

1. Prepare and validate an object-detection dataset in YOLO format.
2. Understand and apply YOLOv8 for brain MRI tumor detection.
3. Fine-tune a pretrained object-detection model on a custom dataset.
4. Understand and evaluate detection metrics such as IoU, Precision, Recall, and mAP.
5. Analyze model performance and identify detection errors.
6. Track experiments and compare different training configurations using MLflow.
7. Register and manage the best-performing model using the MLflow Model Registry.
8. Expose the selected model through a REST API.
9. Build an interactive interface for testing MRI images.
10. Automate the Machine Learning workflow using Apache Airflow.
11. Maintain reproducibility through fixed configurations, documented dependencies, and organized code.

---

## 3. Technologies & Tools

| Technology | Purpose |
|---|---|
| Python | Main programming language |
| YOLOv8 / Ultralytics | Object detection and model training |
| PyTorch | Deep Learning framework used by YOLOv8 |
| OpenCV | Image loading, validation, and processing |
| NumPy | Numerical operations |
| scikit-learn | Dataset splitting and supporting evaluation utilities |
| Kaggle | Dataset source |
| KaggleHub | Dataset downloading |
| Vast.ai | Cloud GPU environment |
| MLflow | Experiment tracking and model registry |
| FastAPI | Model inference REST API |
| Streamlit | Interactive user interface |
| Apache Airflow | Workflow orchestration |
| SQLite | Local development metadata storage |
| Git & GitHub | Version control and project hosting |
| Jira | Project planning and task management |

### Development Environment

The Part 3 development environment was prepared with the following configuration:

| Component | Version / Configuration |
|---|---|
| Python | 3.12.15 |
| PyTorch | 2.14.1+cu130 |
| Ultralytics | 8.4.189 |
| Apache Airflow | 2.10.5 |
| GPU | NVIDIA RTX 3060 Ti – 8 GB VRAM |
| Environment management | uv |
| Operating system | Linux – Vast.ai |

Other library versions should be checked against the project's dependency files before reproducing an experiment.

---

## 4. Dataset

### 4.1 Dataset Source

The project uses the Kaggle dataset:

**MRI for Brain Tumor with Bounding Boxes**

Dataset identifier:

`ahmedsorour1/mri-for-brain-tumor-with-bounding-boxes`

The dataset contains brain MRI images with object-detection annotations.

Each annotation describes a bounding box around an object and its corresponding class.

### 4.2 Detection Classes

The model is configured to detect four classes:

| Class ID | Class Name |
|---:|---|
| 0 | Glioma |
| 1 | Meningioma |
| 2 | No Tumor |
| 3 | Pituitary |

The class names and IDs must remain consistent across dataset annotations, `data.yaml`, training, evaluation, and inference.

### 4.3 YOLO Annotation Format

YOLO uses one text annotation file (`.txt`) for each image.

Each line represents one object:

```text
class_id x_center y_center width height
```

The coordinates are normalized between 0 and 1.

For example:

```text
0 0.512 0.463 0.280 0.315
```

This represents:

- `0`: Object class ID.
- `0.512`: Bounding box center on the horizontal axis.
- `0.463`: Bounding box center on the vertical axis.
- `0.280`: Bounding box width.
- `0.315`: Bounding box height.

The normalized coordinates make annotations independent of the original image resolution.

### 4.4 Dataset Organization

The processed dataset follows the YOLO directory structure:

```text
brain_mri_yolo/
│
├── images/
│   ├── train/
│   ├── val/
│   └── test/
│
├── labels/
│   ├── train/
│   ├── val/
│   └── test/
│
└── data.yaml
```

Each image must have a corresponding annotation file with the same filename.

For example:

```text
images/train/image001.jpg
labels/train/image001.txt
```

An empty annotation file can represent an image containing no annotated object, provided that this interpretation is confirmed for the dataset.

### 4.5 Dataset Configuration

The `data.yaml` file tells YOLO where the images are located and which classes must be detected.

Example:

```yaml
path: /path/to/brain_mri_yolo

train: images/train
val: images/val
test: images/test

nc: 4

names:
  0: Glioma
  1: Meningioma
  2: No Tumor
  3: Pituitary
```

The actual dataset path is generated by the preprocessing pipeline.

---

## 5. Data Exploration & Preprocessing

Before training, the dataset must be inspected, validated, and organized.

### 5.1 Dataset Exploration

The exploration stage focuses on understanding the dataset:

- Inspect the dataset directory structure.
- Identify available splits.
- Examine image extensions and dimensions.
- Read YOLO annotation files.
- Display sample images with their bounding boxes.
- Display class names and bounding-box locations.
- Analyze the distribution of images and annotations per class.
- Identify possible class imbalance.

At least nine annotated images should be visualized to verify that the bounding boxes correspond to the correct objects.

### 5.2 Image Validation

The preprocessing stage checks the image files before using them for training.

Supported extensions:

- `.jpg`
- `.jpeg`
- `.png`
- `.bmp`

The validation process checks:

- Whether an image can be opened.
- Whether the image is corrupted.
- Whether the corresponding annotation exists.
- Whether the image and annotation filenames match.

Invalid or unsupported files should be excluded and counted.

### 5.3 Annotation Validation

Each non-empty annotation line must contain exactly five values:

```text
class_id x_center y_center width height
```

The validation process checks:

- The class ID is an integer.
- The class ID belongs to the four configured classes.
- Coordinates are normalized between 0 and 1.
- Bounding-box width and height are positive.
- Every annotation line follows the expected format.

### 5.4 Dataset Splitting

The objective is to maintain three independent subsets:

- **Training:** Used to train model weights.
- **Validation:** Used to monitor performance and select configurations.
- **Testing:** Reserved for final evaluation.

Images and their corresponding annotation files must always remain together during splitting.

A fixed random seed is used for reproducibility.

The preprocessing implementation uses the available source splits and separates a portion of the original training data for testing. The exact final split counts must be verified after preprocessing.

### 5.5 Preprocessing Output

The preprocessing module is responsible for:

1. Downloading or locating the original dataset.
2. Finding the source image and annotation folders.
3. Validating images and labels.
4. Collecting valid image-annotation pairs.
5. Splitting the data.
6. Organizing images and labels into YOLO directories.
7. Generating `data.yaml`.
8. Checking the resulting dataset structure.

The prepared implementation also introduces a staging and backup approach to reduce the risk of replacing an existing processed dataset before the new one has been validated. This behavior still requires testing on the target environment.

---

## 6. Data Augmentation

Data augmentation helps the model generalize to variations in MRI images.

The augmentation strategy should remain conservative because medical images contain important anatomical information.

Potential transformations include:

| Transformation | Purpose |
|---|---|
| Small rotation | Simulate slight image orientation changes |
| Translation | Handle small shifts in object position |
| Scaling | Improve robustness to object-size variations |
| Brightness adjustment | Handle intensity variations |

Some transformations require particular attention:

- **Horizontal flipping:** Must be justified because the anatomical side of the brain can matter.
- **Strong rotation:** Can produce unrealistic MRI orientations.
- **Hue and saturation:** Should be limited because MRI images are generally grayscale.
- **Mosaic:** Should be evaluated carefully because combining images can change the context and appearance of tumor regions.

Augmentation must be applied only to training data. Validation and test images must not be augmented.

The final augmentation settings should be documented with the corresponding experiment.

---

## 7. YOLOv8 Model Training

### 7.1 Why YOLOv8?

YOLO (You Only Look Once) is an object-detection architecture designed to detect objects and predict their locations in images.

YOLOv8 provides:

- Object localization.
- Class prediction.
- Confidence scores.
- Bounding-box predictions.
- Training and validation tools.
- Detection metrics and visualizations.

For this project, YOLOv8 is adapted to detect the four configured brain MRI classes.

### 7.2 Transfer Learning

Instead of training the detector entirely from random weights, the project starts from a pretrained YOLOv8 model.

The initial model is:

```text
yolov8n.pt
```

The `n` version (Nano) is selected as a lightweight starting point, considering the available GPU resources.

The pretrained weights are adapted to the custom brain MRI detection dataset.

### 7.3 Training Configuration

The main configuration is centralized in:

```text
src/config.py
```

Current baseline configuration:

| Parameter | Value |
|---|---:|
| Model | `yolov8n.pt` |
| Image size | 224 |
| Batch size | 16 |
| Epochs | 50 |
| Patience | 10 |
| Random state | 42 |
| Confidence threshold | 0.25 |

These values represent the prepared baseline configuration and should be adjusted only through documented experiments.

### 7.4 Training Process

The training module:

1. Checks whether the dataset configuration exists.
2. Detects whether a CUDA-compatible GPU is available.
3. Loads the pretrained YOLOv8 model.
4. Starts the training timer.
5. Trains using the configured dataset and hyperparameters.
6. Measures the training duration.
7. Locates the best model weights.

The main training artifacts are:

```text
best.pt
last.pt
```

- `best.pt`: Weights corresponding to the best validation performance during training.
- `last.pt`: Weights from the last completed training epoch.

### 7.5 Training Monitoring

Training outputs should be inspected to understand model behavior.

Important outputs include:

- Training and validation losses.
- Precision and Recall curves.
- mAP metrics.
- Confusion matrix.
- `results.csv`.
- Training duration.
- Best and last model weights.

If training performance continues improving while validation performance stagnates or deteriorates, the model may be overfitting.

Early stopping through patience helps prevent unnecessary training.

---

## 8. Model Evaluation

Object detection requires evaluating both classification quality and bounding-box localization.

### 8.1 Intersection over Union (IoU)

IoU measures the overlap between a predicted bounding box and the ground-truth bounding box.

\[
IoU = \frac{Area\ of\ Intersection}{Area\ of\ Union}
\]

A higher IoU indicates a better overlap between the predicted and actual object locations.

### 8.2 Precision

Precision measures how many predicted detections are correct.

\[
Precision = \frac{TP}{TP + FP}
\]

Where:

- TP: True Positives.
- FP: False Positives.

High precision means fewer incorrect detections.

### 8.3 Recall

Recall measures how many actual objects were successfully detected.

\[
Recall = \frac{TP}{TP + FN}
\]

Where:

- TP: True Positives.
- FN: False Negatives.

In a medical-support context, missed detections are particularly important to analyze.

### 8.4 Mean Average Precision (mAP)

The project uses two main detection metrics:

- **mAP@0.5:** Mean Average Precision at an IoU threshold of 0.5.
- **mAP@0.5:0.95:** Mean Average Precision averaged across multiple IoU thresholds from 0.5 to 0.95.

The second metric is stricter because it evaluates localization quality at multiple overlap thresholds.

### 8.5 Error Analysis

The evaluation process should include:

- Global Precision and Recall.
- Global mAP@0.5.
- Global mAP@0.5:0.95.
- Per-class metrics.
- Precision-Recall curves.
- F1 curves.
- Confusion matrix.
- Correct detections.
- False Positives.
- False Negatives.
- Examples of poorly localized bounding boxes.

The model should be evaluated on validation data during experimentation.

The test set must be reserved for the final evaluation after model selection and should be evaluated only once.

**Important:** The current prepared Airflow evaluation task uses the validation split for model comparison. Final test-set evaluation remains a separate step to complete and document.

---

## 9. Experiment Tracking with MLflow

### 9.1 Why MLflow?

MLflow is used to organize, track, and compare Machine Learning experiments.

Instead of relying only on manually saved files, each training experiment is associated with a tracked MLflow run.

### 9.2 Experiment Structure

Each training configuration should create a separate run.

For every run, the project aims to track:

**Parameters**

- Model name.
- Number of epochs.
- Image size.
- Batch size.
- Patience.
- Other experiment settings when applicable.

**Metrics**

- Precision.
- Recall.
- mAP@0.5.
- mAP@0.5:0.95.
- Training duration.

**Artifacts**

- Best YOLO weights.
- Training outputs.
- Evaluation visualizations.
- Relevant configuration files.

### 9.3 Experiment Comparison

At least two experiments must be performed.

The experiments should differ by only one configuration setting to make the comparison meaningful.

Possible comparisons:

| Experiment | Model | Difference |
|---|---|---|
| Run 1 | YOLOv8n | Baseline |
| Run 2 | YOLOv8s | Model size |

Another possible experiment is to compare two augmentation configurations while keeping the other settings unchanged.

The objective is to identify the best model without comparing experiments that changed many parameters simultaneously.

### 9.4 Model Selection

The model should not be selected based only on one metric.

The selection should consider:

- Recall.
- Precision.
- mAP@0.5.
- mAP@0.5:0.95.
- Detection quality per class.

For medical-support applications, Recall deserves particular attention because false negatives represent missed detections. However, Recall should not be increased at the expense of an unacceptable loss in overall detection quality.

### 9.5 MLflow Model Registry

The Model Registry is used to manage model versions.

The selected model receives the alias:

```text
champion
```

The alias identifies the currently selected model version.

When a new experiment produces a better model, the alias can be moved to the new version.

This allows downstream services to use the selected model without hardcoding a specific version number.

---

## 10. Model Serving with FastAPI

FastAPI is intended to expose the selected YOLOv8 model through a REST API.

The API retrieves the model associated with the MLflow Registry alias:

```text
champion
```

The model should be loaded once when the API starts, rather than reloaded for every prediction request.

### 10.1 Health Endpoint

```http
GET /health
```

The health endpoint checks whether:

- The API is running.
- The model is available.
- The inference service is ready.

### 10.2 Prediction Endpoint

```http
POST /predict
```

The prediction endpoint receives a brain MRI image and returns the detection results.

The response should contain:

- Predicted class.
- Confidence score.
- Bounding-box coordinates.

Example response structure:

```json
{
  "detections": [
    {
      "class_id": 0,
      "class_name": "Glioma",
      "confidence": 0.94,
      "bbox": {
        "x1": 120,
        "y1": 85,
        "x2": 260,
        "y2": 210
      }
    }
  ],
  "message": "Detection completed successfully."
}
```

This is an illustrative response format; the final API implementation must define and follow its actual response schema.

### 10.3 Input Validation

The API should validate:

- Uploaded file presence.
- Supported image extensions.
- Image readability.
- Optional confidence threshold.

Supported formats:

```text
.jpg
.jpeg
.png
.bmp
```

The API should return clear errors for invalid input or unavailable services.

Typical HTTP responses include:

| Status | Meaning |
|---|---|
| 200 | Prediction completed |
| 400 | Invalid request or unsupported input |
| 422 | Invalid request data |
| 500 | Internal server error |

If no detection is returned, the API should clearly communicate:

```text
No tumor detected.
```

### 10.4 API Documentation

FastAPI automatically provides interactive API documentation.

Once the API is running, the documentation is available at:

```text
/docs
```

This allows developers to inspect and test the available endpoints.

---

## 11. Interactive Interface with Streamlit

Streamlit provides a simple interface for testing the detection model.

The interface communicates with FastAPI instead of loading the YOLO model directly.

This separation keeps the model inference logic inside the API.

### 11.1 Main Features

The interface should provide:

1. MRI image upload.
2. Image preview.
3. Confidence threshold selection.
4. Prediction request to FastAPI.
5. Display of detected bounding boxes.
6. Display of predicted classes.
7. Display of confidence scores.
8. Clear message when no tumor is detected.
9. Error messages when the API is unavailable or the image is invalid.

### 11.2 Application Architecture

```text
                  User
                   |
                   v
          Streamlit Interface
                   |
                   | HTTP POST /predict
                   v
              FastAPI
                   |
                   v
          YOLOv8 Champion Model
                   |
                   v
           Detection Results
                   |
                   v
          Streamlit Visualization
```

Streamlit is responsible for interaction and visualization, while FastAPI handles inference.

---

## 12. Pipeline Automation with Apache Airflow

Apache Airflow is used to orchestrate the Machine Learning workflow.

Instead of manually executing every stage, Airflow organizes the process into tasks with dependencies.

### 12.1 Pipeline Architecture

The main pipeline contains four sequential tasks:

```text
             Start
               |
               v
         Preprocessing
               |
               v
           Training
               |
               v
          Evaluation
               |
               v
          Registration
               |
               v
              End
```

### 12.2 Pipeline Tasks

| Task | Responsibility |
|---|---|
| `preprocessing` | Prepare and validate the YOLO dataset |
| `training` | Train YOLOv8 and track the experiment |
| `evaluation` | Evaluate the trained model and log metrics |
| `registration` | Register the model and compare it with the current champion |

### 12.3 Task Dependencies

The tasks execute sequentially:

```python
preprocessing >> training >> evaluation >> registration
```

This means that a task starts only after its previous task succeeds.

### 12.4 Retries and Error Handling

The DAG is configured with retry behavior to allow failed tasks to run again.

The prepared DAG configuration includes:

- One retry.
- A five-minute retry delay.
- No automatic historical catchup.
- A maximum of one active pipeline run.

Airflow provides a graphical interface for monitoring task states, execution logs, and pipeline progress.

### 12.5 Champion Model Update

The registration task compares the new model against the currently registered champion.

The intended selection rule considers both Recall and mAP@0.5:0.95.

The champion alias should move only when the new model satisfies the model-selection criteria.

This prevents an inferior experiment from automatically replacing the selected model.

---

## 13. Project Architecture

The project separates configuration, preprocessing, training, orchestration, and application components.

```text
NeuroVision/
│
├── api/
│   └── FastAPI application
│
├── app/
│   └── Streamlit application
│
├── dags/
│   └── neurovision_pipeline.py
│
├── data/
│   ├── raw/
│   │   └── Original dataset
│   │
│   └── processed/
│       └── brain_mri_yolo/
│           ├── images/
│           │   ├── train/
│           │   ├── val/
│           │   └── test/
│           │
│           ├── labels/
│           │   ├── train/
│           │   ├── val/
│           │   └── test/
│           │
│           └── data.yaml
│
├── models/
│   └── Model-related files
│
├── notebooks/
│   └── Data exploration and experiments
│
├── src/
│   ├── config.py
│   ├── preprocessing.py
│   ├── train.py
│   └── airflow_tasks.py
│
├── weights/
│   └── Model weights
│
├── runs/
│   └── YOLO training outputs
│
├── mlartifacts/
│   └── MLflow artifacts
│
├── mlruns/
│   └── MLflow local experiment data
│
├── airflow/
│   └── Airflow configuration and metadata
│
├── start_airflow.sh
├── start_mlflow.sh
│
├── yolov8n.pt
├── mlflow.db
├── pyproject.toml
├── uv.lock
├── .gitignore
└── README.md
```

Some directories are created or populated during execution. Dataset images, training outputs, and model artifacts are not necessarily committed to GitHub.

### 13.1 Main Source Files

#### `src/config.py`

Centralizes the project configuration:

- Dataset paths.
- Model name.
- Class names.
- Number of classes.
- Supported extensions.
- Random seed.
- Image size.
- Batch size.
- Epochs.
- Patience.
- Confidence threshold.
- Augmentation parameters.

#### `src/preprocessing.py`

Responsible for:

- Loading the dataset.
- Discovering image and annotation folders.
- Validating images.
- Validating YOLO annotations.
- Preparing dataset splits.
- Organizing the final directory structure.
- Creating `data.yaml`.

#### `src/train.py`

Responsible for:

- Detecting the available training device.
- Loading pretrained YOLOv8 weights.
- Training the model.
- Measuring training duration.
- Locating the best model weights.

#### `src/airflow_tasks.py`

Contains the Python functions called by Airflow:

- `preprocessing_task()`
- `training_task()`
- `evaluation_task()`
- `registration_task()`

It connects the individual Machine Learning stages to MLflow and Airflow.

#### `dags/neurovision_pipeline.py`

Defines the Airflow DAG, its Python operators, configuration, and task dependencies.

#### `api/`

Contains the FastAPI application responsible for loading the selected model and providing prediction endpoints.

#### `app/`

Contains the Streamlit interface used to upload MRI images and visualize prediction results.

---

## 14. Installation & Setup

### 14.1 Clone the Repository

```bash
git clone https://github.com/ISMAILBENMEZI/NeuroVision-AI-Part-3.git

cd NeuroVision
```

Replace the repository URL if the project is hosted under a different GitHub repository.

### 14.2 Python Environment

The project uses Python 3.12 and `uv` for environment and dependency management.

The dependency definitions are maintained in:

```text
pyproject.toml
uv.lock
```

Create or synchronize the environment according to the dependency configuration:

```bash
uv sync
```

**Note:** Airflow is sensitive to Python and dependency versions. The prepared Vast.ai environment uses Airflow 2.10.5. Avoid unintentionally replacing its working environment with a different dependency resolution.

### 14.3 GPU Verification

Check GPU availability:

```bash
nvidia-smi
```

Check PyTorch CUDA availability:

```python
import torch

print(torch.cuda.is_available())
print(torch.cuda.get_device_name(0))
```

The project should use the GPU when CUDA is available and fall back to the CPU otherwise.

### 14.4 Dataset Preparation

The preprocessing module is responsible for preparing the dataset.

The resulting `data.yaml` must exist before training starts.

### 14.5 Start MLflow

The project includes:

```text
start_mlflow.sh
```

Use the prepared script to start the MLflow tracking service.

The configured local tracking database is:

```text
mlflow.db
```

Tracking URI:

```text
sqlite:////home/user/NeuroVision/mlflow.db
```

MLflow stores experiment metadata and associated artifacts.

### 14.6 Start Airflow

The project includes:

```text
start_airflow.sh
```

Make the script executable:

```bash
chmod +x start_airflow.sh
```

Start Airflow:

```bash
./start_airflow.sh
```

The prepared configuration uses:

```text
AIRFLOW_HOME=/home/user/NeuroVision/airflow
```

```text
DAG_FOLDER=/home/user/NeuroVision/dags
```

Airflow web interface:

```text
http://localhost:8085
```

The scheduler and webserver must use the same Airflow environment and configuration.

### 14.7 Run the Pipeline

After verifying the Airflow environment and registering the DAG, the pipeline can be triggered from the Airflow interface.

The DAG ID is:

```text
neurovision_pipeline
```

The tasks execute in the defined order.

### 14.8 Run the API and Interface

The FastAPI and Streamlit applications should be started separately after their entry-point files and dependencies are finalized.

The expected execution flow is:

```text
Streamlit → FastAPI → MLflow Champion Model → YOLOv8 Inference
```

The exact launch commands depend on the final application entry points.

---

## 15. Reproducibility

Reproducibility is important when comparing Machine Learning experiments.

The project uses a centralized configuration file and a fixed random seed.

To reproduce an experiment, document:

- Dataset source and version.
- Dataset split.
- Class names and IDs.
- YOLOv8 model variant.
- Image size.
- Batch size.
- Epochs.
- Patience.
- Augmentation settings.
- Python version.
- PyTorch version.
- Ultralytics version.
- GPU configuration.
- MLflow run ID.

Every experiment should be identifiable through MLflow.

Model selection should be based on validation metrics, with the test set reserved for final evaluation.

---

## 16. Expected Results & Evaluation Reports

The project should produce the following outputs after successful training and evaluation:

| Output | Description |
|---|---|
| `best.pt` | Best trained YOLOv8 weights |
| `last.pt` | Last training checkpoint |
| `results.csv` | Training and validation metrics |
| Confusion matrix | Classification behavior |
| Precision curve | Precision across confidence thresholds |
| Recall curve | Recall across confidence thresholds |
| F1 curve | F1 score across confidence thresholds |
| PR curve | Precision-Recall relationship |
| Detection examples | Predicted and ground-truth bounding boxes |
| MLflow runs | Experiment history and metrics |
| Champion model | Selected registered model version |

Actual numerical results should be added here after the final experiments have been executed.

No performance values are reported in this README until they have been measured and verified.

---

## 17. Project Development Status

NeuroVision AI Part 3 is being developed incrementally.

The following table distinguishes prepared components from work that still requires validation.

| Component | Status |
|---|---|
| Project configuration | Prepared |
| Dataset preprocessing | Implemented, requires testing |
| YOLOv8 training module | Prepared |
| MLflow training tracking | Prepared, requires verification |
| Validation evaluation | Prepared, requires testing |
| Model Registry | Prepared, requires integration testing |
| Airflow task separation | Prepared |
| Airflow DAG | Prepared, UI and execution require verification |
| Airflow startup script | Prepared, requires testing |
| FastAPI inference service | Requires completion and testing |
| Streamlit interface | Requires completion and testing |
| Final test-set evaluation | Pending |
| Two-run experiment comparison | Pending |
| Final results documentation | Pending |

This status reflects the development state of the project documentation and prepared code, not a claim that the complete pipeline has already passed end-to-end testing.

---

## 18. Future Improvements

Possible improvements include:

- Compare YOLOv8n with YOLOv8s.
- Experiment with different image sizes.
- Optimize augmentation settings.
- Improve detection performance for underrepresented classes.
- Add more detailed error analysis.
- Improve model selection criteria.
- Complete automated champion model deployment.
- Add API testing and validation.
- Add automated tests for preprocessing.
- Improve monitoring and logging.
- Add deployment configuration for a production-like environment.
- Evaluate inference speed and resource consumption.

---

## 19. Limitations

This project has several important limitations:

1. The quality of the predictions depends on the dataset and annotation quality.
2. A model trained on one dataset may not generalize to MRI images from different hospitals or scanners.
3. Detection performance does not establish clinical validity.
4. Confidence scores are not medical probabilities.
5. False positives and false negatives must be carefully analyzed.
6. The model requires external validation before any real-world medical use.
7. GPU memory and computing resources can limit model size and training configuration.

NeuroVision AI is an educational Machine Learning project and must not be used as an autonomous diagnostic system.

---

## 20. Project Planning

The development process is organized into major work areas:

| Epic | Main Activities |
|---|---|
| Dataset Preparation | Exploration, validation, cleaning, splitting |
| Object Detection | YOLOv8 configuration and training |
| Model Evaluation | Metrics, visual analysis, error inspection |
| Experiment Tracking | MLflow runs and experiment comparison |
| Model Management | Registry, versions, champion alias |
| API Development | FastAPI endpoints and validation |
| User Interface | Streamlit image upload and visualization |
| Workflow Automation | Airflow DAG and task orchestration |
| Documentation | README, reproducibility, and results |

Jira is used to organize the work into Epics and smaller implementation tasks.

---

## 21. Author

**Ismail Ben Mezi**

Second-year student at **YouCode – UM6P**, specializing in Artificial Intelligence after completing a first year in Full Stack Development.

Areas of interest:

- Artificial Intelligence
- Machine Learning
- Deep Learning
- Computer Vision
- Full Stack Development

GitHub: [ISMAILBENMEZI](https://github.com/ISMAILBENMEZI)

Portfolio: [Personal Portfolio](https://ismailbenmezi.github.io/Portfolio/)

---

## 22. Acknowledgments

- [Ultralytics](https://github.com/ultralytics/ultralytics) – YOLOv8 object detection framework.
- [Kaggle](https://www.kaggle.com/) – Brain MRI dataset.
- [MLflow](https://mlflow.org/) – Experiment tracking and model management.
- [Apache Airflow](https://airflow.apache.org/) – Workflow orchestration.
- [FastAPI](https://fastapi.tiangolo.com/) – API development.
- [Streamlit](https://streamlit.io/) – Interactive application development.
- [OpenCV](https://opencv.org/) – Image processing.
- [Vast.ai](https://vast.ai/) – Cloud GPU infrastructure.

---

**NeuroVision AI – From Brain MRI Classification to Tumor Detection.**