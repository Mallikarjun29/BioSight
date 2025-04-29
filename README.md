# BioSight - Biological Image Classification and Organization

## Overview

BioSight is a comprehensive solution for biological image classification and organization, combining a user-facing web application with an automated drift detection and model retraining pipeline. The system allows users to upload and classify biological images while maintaining model quality through automated checks and updates.

## Features

### Web Application
*   **User Authentication:** Secure registration and login using JWT tokens stored in HttpOnly cookies.
*   **Image Upload:** Supports uploading multiple image files (`.png`, `.jpg`, `.jpeg`).
*   **Automatic Classification:** Classifies uploaded images into biological categories (Amphibia, Animalia, Arachnida, Aves, Fungi, Insecta, Mammalia, Mollusca, Plantae, Reptilia).
*   **Image Organization:** Automatically moves classified images into corresponding folders.
*   **Results Visualization:** Displays classified images grouped by predicted class in a tabbed interface.
*   **Classification Correction:** Allows users to manually change the assigned class of an image via a dropdown. Changes are reflected in the file system and database.
*   **Image Deletion:** Users can delete uploaded images, removing them from the file system and database.
*   **Download:** Option to download all organized images as a single Zip archive.
*   **Application & System Monitoring:** Includes a `/metrics` endpoint scraped by Prometheus, visualized with a pre-configured Grafana dashboard showing application performance, API usage, and system metrics.
*   **Docker Support:** Includes configuration for running the application, database, Prometheus, and Grafana using Docker Compose.
*   **Health Check:** `/health` endpoint to verify application status, database connection, and model loading.

### Model Building Pipeline
*   **Custom Model Architecture:** Enhanced ResNet50 with customizable dropout and transfer learning configurations.
*   **Transfer Learning:** Multiple freezing strategies for fine-tuning different parts of the model.
*   **Hyperparameter Optimization:** Automated hyperparameter tuning using Hyperopt and MLflow.
*   **Experiment Tracking:** Comprehensive logging of model metrics and artifacts using MLflow.
*   **Validation & Testing:** Structured data splitting and evaluation on separate validation and test sets.

### Drift Detection and Retraining Pipeline
*   **Automated Data Drift Detection:** Monitors incoming image data to detect drift compared to the original training distribution.
*   **Model Retraining:** Automatically retrains the classification model when significant drift is detected.
*   **CI/CD Integration:** Runs as GitHub Actions workflow for automated execution.
*   **Data Versioning:** Uses DVC (Data Version Control) to manage large datasets and models.
*   **Parameterized Configuration:** Uses params.yaml for configurable pipeline settings.

## File Structure

```
BioSight/
├── client-side/            # Web application components
│   ├── biosight/            # FastAPI application source code
│   │   ├── __init__.py
│   │   ├── app.py           # Main FastAPI application logic and routes
│   │   ├── static/          # Static files (CSS, JS)
│   │   ├── templates/       # Jinja2 HTML templates
│   │   ├── routes/          # API route modules
│   │   ├── utils/           # Utility modules
│   │   └── models/          # Directory for ML model files
│   ├── grafana/             # Grafana configuration
│   ├── prometheus/          # Prometheus configuration
│   ├── uploads/             # Default folder for temporary uploads
│   ├── organized_images/    # Default folder for classified images
│   ├── docker-compose.yml   # Defines services for the web application
│   ├── Dockerfile           # Instructions to build the FastAPI application image
│   └── requirements.txt     # Python package dependencies for the web application
│
├── model_building_pipeline/ # Model building components
│   ├── model.py             # ResNet model architecture definition
│   ├── train.py             # Training script with command-line arguments
│   ├── experiment.py        # Hyperparameter optimization script
│   ├── prepare_data.py      # Data loading and preprocessing utilities
│   └── requirements.txt     # Python package dependencies for model building
│
├── drift_detection_and_retraining_pipeline/  # Drift detection pipeline components
│   ├── drift_detection/     # Drift detection-specific code
│   │   ├── check_drift_batch.py    # Script to detect drift in recent images
│   │   ├── prepare_data.py         # Data preprocessing for drift detection
│   │   ├── train_drift.py          # Training script for drift detector model
│   │   └── drift_models/           # Directory for drift detection models
│   ├── prepare_retrain_data.py    # Prepares data for model retraining
│   ├── retrain.py                 # Retrains the main classification model
│   ├── evaluate.py                # Evaluates the retrained model
│   ├── requirements.txt           # Python package dependencies for the pipeline
│   ├── dvc.yaml                   # DVC pipeline definition
│   ├── dvc.lock                   # DVC pipeline state record
│   ├── params.yaml                # Parameters for the pipeline stages
│   └── inaturalist_12K/           # Dataset used for training/validation (DVC tracked)
│       ├── train/                 # Training data
│       └── val/                   # Validation data
│
├── .github/                 # GitHub-specific configuration
│   └── workflows/           # GitHub Actions workflow definitions
│       └── drift_pipeline.yml  # Workflow for drift detection and retraining
│
├── .dvc/                    # DVC configuration directory
├── .gitignore               # Specifies intentionally untracked files for Git
├── README.md                # This file
└── inaturalist_12K.dvc      # DVC tracking file for the dataset
```

## Dependencies

### External Services:

*   **Docker & Docker Compose:** For containerization and multi-container management.
*   **MongoDB:** NoSQL database for storing metadata.
*   **Prometheus & Grafana:** For monitoring and visualization.
*   **Node Exporter:** For exposing host system metrics.
*   **GitHub:** For source code hosting and GitHub Actions for CI/CD.
*   **MLflow:** For experiment tracking and model registry.
*   **DVC Remote Storage:** (e.g., Google Drive, S3) for storing large data files and models.
*   **Pre-trained Model:** Classification model weights file (`best_model_resnet.pth`).

### Python Packages:

See respective requirements.txt files in the project directories.

## Setup and Installation

### 1. Web Application Setup

#### Option 1: Using Docker (Recommended)

1.  **Clone the Repository:**
    ```bash
    git clone <repository-url>
    cd BioSight/client-side
    ```

2.  **Prepare Environment Variables:**
    ```bash
    cp .env.example .env
    ```
    Edit the `.env` file and configure your settings (especially `SECRET_KEY`).

3.  **Place Model Weights:**
    ```bash
    mkdir -p biosight/models/weights/
    # Copy your best_model_resnet.pth file into the directory above
    ```

4.  **Build and Run with Docker Compose:**
    ```bash
    docker-compose up --build -d
    ```

5.  **Access Services:**
    *   **BioSight Application:** `http://localhost:8000`
    *   **Grafana:** `http://localhost:3000` (Default user/pass: admin/admin)
    *   **Prometheus:** `http://localhost:9090`

6.  **Stopping the Application Stack:**
    ```bash
    docker-compose down  # Add -v to also remove volumes
    ```

#### Option 2: Manual Local Setup

1.  **Clone the Repository** and navigate to the client-side directory.
2.  **Install MongoDB** locally.
3.  **Create and Activate a Virtual Environment:**
    ```bash
    python3 -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    ```
4.  **Install Dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
5.  **Configure Environment Variables** in a `.env` file.
6.  **Place Model Weights** in the appropriate directory.
7.  **Run the Application:**
    ```bash
    uvicorn biosight.app:app --reload --host 0.0.0.0 --port 8000
    ```
8.  **Access the Application:** `http://localhost:8000`

### 2. Model Building Pipeline Setup

The model building pipeline creates and trains the deep learning models used for biological image classification.

#### Initial Setup

1. **Navigate to the Model Building Directory:**
   ```bash
   cd model_building_pipeline
   ```

2. **Create and Activate a Virtual Environment:**
   ```bash
   python3 -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install Required Dependencies:**
   ```bash
   pip install torch torchvision mlflow hyperopt scikit-learn numpy
   # Or use the requirements.txt file if available:
   # pip install -r requirements.txt
   ```

4. **Setup MLflow Tracking (Optional but Recommended):**
   ```bash
   # Start MLflow tracking server
   mlflow server --backend-store-uri sqlite:///mlflow.db --default-artifact-root ./mlflow-artifacts
   ```
   Access the MLflow UI at `http://localhost:5000`

#### Training a Model

1. **Basic Model Training:**
   ```bash
   python train.py --experiment_name resnet_training --epochs 10 --batch_size 32 --learning_rate 0.001 --freeze_strategy upto_stage_3
   ```

2. **Available Training Arguments:**
   - `--experiment_name`: MLflow experiment name
   - `--epochs`: Number of training epochs
   - `--batch_size`: Batch size for training
   - `--learning_rate`: Learning rate for the optimizer
   - `--freeze_strategy`: Which layers to freeze (`none`, `upto_stage_1`, `upto_stage_2`, `upto_stage_3`)
   - `--dropout_rate`: Dropout rate in the classifier head

3. **Hyperparameter Optimization:**
   ```bash
   python experiment.py
   ```
   This will start hyperparameter tuning using Hyperopt with MLflow tracking.

#### Using Trained Models

After training, models are saved in two formats:
- PyTorch state dict: `best_model_resnet.pth`
- MLflow model registry: Access via the MLflow UI

To use a trained model in the web application, copy the `.pth` file to the web app's model directory:
```bash
cp best_model_resnet.pth ../client-side/biosight/models/weights/
```

#### Model Architecture

The classification model is based on ResNet50 with customizable features:

- Pre-trained weights from ImageNet
- Customizable freezing strategies for transfer learning
- Custom classifier head with dropout for regularization
- 10 output classes for biological classification

### 3. Drift Detection and Retraining Pipeline Setup

#### Initial Setup

1.  **Install DVC:**
    ```bash
    pip install dvc  # Or pip3 install dvc
    ```

2.  **Configure DVC Remote Storage:**
    ```bash
    # Example for Google Drive
    dvc remote add -d myremote gdrive://<your-gdrive-folder-id>
    # Follow authentication steps when prompted
    
    # Or for local storage (not recommended for production)
    dvc remote add -d mylocal /path/to/storage/location
    ```

3.  **Obtain the Dataset:**
    *   You need the inaturalist_12K dataset with `train/` and `val/` subdirectories.
    *   Place this in the repository root, or use DVC to pull it:
        ```bash
        dvc pull inaturalist_12K.dvc
        ```

4.  **Install Pipeline Dependencies:**
    ```bash
    cd drift_detection_and_retraining_pipeline
    pip install -r requirements.txt
    ```

#### Running the Pipeline Locally

You can manually run the drift detection and retraining pipeline locally:

1.  **Navigate to Pipeline Directory:**
    ```bash
    cd drift_detection_and_retraining_pipeline
    ```

2.  **Run the DVC Pipeline:**
    ```bash
    dvc repro
    ```
    This will execute stages defined in dvc.yaml in the correct order.

3.  **Check Results:**
    *   Drift detection reports will be in the `drift_detection/` directory
    *   Updated models will be in appropriate model directories based on params.yaml

#### Pipeline Parameters

The pipeline is configured using params.yaml. Key parameters include:

*   **Data Settings:**
    ```yaml
    data_dir: ../inaturalist_12K  # Path to the dataset relative to pipeline dir
    batch_size: 32                # Batch size for training
    ```

*   **Training Settings:**
    ```yaml
    epochs: 1                     # Number of epochs for training
    learning_rate: 0.001          # Learning rate
    freeze_strategy: upto_stage_3 # Feature extraction strategy for transfer learning
    ```

*   **Thresholds:**
    ```yaml
    drift_threshold: 0.05         # Threshold for detecting drift
    min_samples_for_retraining: 50 # Minimum drifted samples needed for retraining
    ```

Adjust these values in params.yaml to customize the pipeline behavior.

### 4. GitHub Actions Setup for Automated Pipeline

The drift detection and retraining pipeline can be triggered automatically via GitHub Actions.

1.  **Ensure GitHub Repository Secrets:**
    *   If your DVC remote requires authentication (e.g., GDrive), set up the necessary secrets:
        *   Go to your GitHub repository → Settings → Secrets and Variables → Actions
        *   Add secrets like `DVC_GDRIVE_CREDENTIALS_DATA` with your credentials

2.  **Push Your Code and DVC Files:**
    ```bash
    # Add and commit modified parameters or code
    git add drift_detection_and_retraining_pipeline/params.yaml
    git add .github/workflows/drift_pipeline.yml
    git commit -m "Update pipeline configuration"
    
    # Add and commit DVC tracking files (NOT the actual data)
    git add *.dvc
    git add .dvc/config
    git commit -m "Update DVC configuration"
    
    # Push to GitHub
    git push origin main
    ```

3.  **Verify Workflow Execution:**
    *   Go to your GitHub repository → Actions tab
    *   You should see the workflow "Drift Detection and Retraining Pipeline" running or queued
    *   Check the logs for any issues

4.  **Workflow Triggers:**
    *   **Automatic:** The workflow runs automatically on push to the main branch (configurable in `drift_pipeline.yml`)
    *   **Manual:** You can also trigger the workflow manually via the GitHub Actions UI using the "workflow_dispatch" event

## Usage

### Web Application Usage

1.  **Register/Login** at `http://localhost:8000`
2.  **Upload Images** using the form on the main page
3.  **View and Manage Results** on the results page
    *   Use tabs to filter by class
    *   Use dropdowns to correct classifications
    *   Delete images with the '×' button
    *   Download all classified images as a zip

### Model Building Workflow

1.  **Data Preparation:**
    *   Ensure your dataset follows the expected structure with class-based subdirectories
    *   Run `python prepare_data.py` to validate dataset structure and preview data loading

2.  **Model Training:**
    *   For basic training, use `python train.py` with appropriate arguments
    *   For hyperparameter optimization, use `python experiment.py`

3.  **Experiment Analysis:**
    *   View training results and compare experiments in the MLflow UI
    *   Select the best model based on validation metrics

4.  **Model Deployment:**
    *   Copy the best model file to the web application for inference

### Checking Pipeline Status

1.  **Locally:**
    ```bash
    cd drift_detection_and_retraining_pipeline
    dvc dag      # Show pipeline dependencies
    dvc status   # Check status of tracked files
    ```

2.  **Via GitHub:**
    *   Go to your GitHub repository → Actions tab
    *   Select the "Drift Detection and Retraining Pipeline" workflow
    *   View the run status and logs

### Viewing Drift Reports

After the pipeline runs, it produces reports on detected drift:

*   **Drift Check Report:** Shows statistics on images checked for drift
*   **Retraining Decision:** Indicates whether retraining was triggered
*   **Model Evaluation:** Provides metrics on any newly trained models

Access these reports in the GitHub Actions run logs or in the local directories after running `dvc repro`.

## Configuration

### Web Application Configuration

*   **Environment Variables (`.env`):** Database URI, secret key, etc.
*   **Application Settings (`biosight/utils/config.py`):** Paths, model settings, allowed file types.
*   **Prometheus & Grafana:** Metrics collection and visualization configuration.

### Model Building Configuration

*   **Command-line Arguments:** Configure training with various arguments (see `train.py --help`).
*   **Model Architecture:** Modify model.py to adjust the model architecture and transfer learning settings.
*   **Hyperparameter Search Space:** Edit the search space in experiment.py for optimization.
*   **MLflow Settings:** Configure experiment names and tracking server URIs.

### Pipeline Configuration

*   **Workflow Definition (`.github/workflows/drift_pipeline.yml`):** Defines the GitHub Actions workflow steps, triggers, and environment.
*   **Pipeline Parameters (`drift_detection_and_retraining_pipeline/params.yaml`):** Controls pipeline behavior, thresholds, and training settings.
*   **DVC Configuration (`.dvc/config`):** Defines remote storage locations for data and models.
*   **Pipeline Stages (`drift_detection_and_retraining_pipeline/dvc.yaml`):** Defines the stages, dependencies, and outputs of the pipeline.

## Troubleshooting Common Issues

### Web Application Issues

*   **Model Loading Errors:** Ensure the model weights file is in the correct location.
*   **Database Connection Issues:** Check MongoDB is running and connection string is correct.
*   **Image Processing Errors:** Verify uploaded images are valid and supported formats.

### Model Building Issues

*   **CUDA Out of Memory:** Reduce batch size or model complexity, or train on CPU.
*   **MLflow Connection Errors:** Ensure the MLflow tracking server is running and accessible.
*   **Dataset Not Found:** Verify the dataset path is correct relative to the execution directory.

### Pipeline Issues

*   **Missing Dataset:** Ensure inaturalist_12K is available and correctly structured.
*   **DVC Remote Access Issues:** Verify credentials and connectivity to DVC remote storage.
*   **GitHub Actions Failures:** Check workflow logs for specific error messages.

### Data Path Issues

*   **FileNotFoundError:** Make sure your params.yaml has the correct relative paths for data directories.
*   **DVC Pull Failures:** Ensure your DVC configuration is correct and remote storage is accessible.

## Contributing

To contribute to the project:

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

We recommend following standard Git flow practices and writing clear commit messages.