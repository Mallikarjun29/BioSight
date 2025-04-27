# BioSight - Biological Image Classification and Organization

## Overview

BioSight is a web application built with FastAPI that allows users to upload biological images, automatically classify them using a pre-trained machine learning model (likely PyTorch-based), and organize the images into folders based on their predicted class. Users can register, log in, manage their uploaded images, correct classifications, and download the organized dataset. The application also includes integrated monitoring using Prometheus and Grafana.

## Features

*   **User Authentication:** Secure registration and login using JWT tokens stored in cookies.
*   **Image Upload:** Supports uploading multiple image files (`.png`, `.jpg`, `.jpeg`).
*   **Automatic Classification:** Classifies uploaded images into biological categories (e.g., Mammalia, Plantae, Amphibia, etc.).
*   **Image Organization:** Automatically moves classified images into corresponding folders.
*   **Results Visualization:** Displays classified images grouped by predicted class in a tabbed interface.
*   **Classification Correction:** Allows users to manually change the assigned class of an image.
*   **Image Deletion:** Users can delete uploaded images.
*   **Download:** Option to download all organized images as a single Zip archive.
*   **Application & System Monitoring:** Includes a `/metrics` endpoint scraped by Prometheus, visualized with a pre-configured Grafana dashboard showing application performance (predictions, latency, uploads per user), API usage, and system metrics (CPU, Memory, Disk, Network).
*   **Docker Support:** Includes configuration for running the application, database, Prometheus, and Grafana using Docker Compose.

## File Structure

```
/BioSight/
├── client-side/
│   ├── biosight/            # FastAPI application source code
│   │   ├── __init__.py
│   │   ├── app.py             # Main FastAPI application logic and routes
│   │   ├── static/            # Static files (CSS, JS)
│   │   │   ├── css/
│   │   │   │   └── style.css
│   │   │   └── js/
│   │   │       └── auth.js
│   │   ├── templates/         # Jinja2 HTML templates
│   │   │   ├── index.html
│   │   │   ├── login.html
│   │   │   ├── register.html
│   │   │   └── result.html
│   │   ├── routes/            # API route modules
│   │   │   ├── __init__.py
│   │   │   ├── auth.py
│   │   │   └── user.py
│   │   ├── utils/             # Utility modules
│   │   │   ├── __init__.py
│   │   │   ├── config.py
│   │   │   ├── constants.py
│   │   │   ├── database.py
│   │   │   ├── file_operations.py
│   │   │   ├── image_processor.py
│   │   │   ├── model_loader.py
│   │   │   ├── monitoring.py    # Prometheus metrics definitions
│   │   │   └── security.py
│   │   └── models/            # Directory for ML model files
│   │       └── weights/
│   │           └── model_weights.pth
│   ├── grafana/             # Grafana configuration
│   │   ├── dashboards/        # Dashboard JSON definitions
│   │   │   └── model.json
│   │   └── provisioning/      # Datasource and dashboard provisioning
│   │       ├── dashboards/
│   │       │   └── default.yaml
│   │       └── datasources/
│   │           └── default.yaml
│   ├── prometheus/          # Prometheus configuration
│   │   └── prometheus.yml
│   ├── Dockerfile             # Instructions to build the FastAPI application image
│   ├── docker-compose.yml     # Defines services (app, db, prometheus, grafana, node-exporter)
│   ├── requirements.txt       # Python package dependencies
│   ├── README.md              # This file
│   └── .env.example           # Example environment variables file
```

## Dependencies

### External Services:
*   **Docker:** Required for containerization. Install Docker Desktop (Windows/Mac) or Docker Engine (Linux).
*   **Docker Compose:** Usually included with Docker Desktop, or install separately on Linux.
*   **MongoDB:** A running MongoDB instance is required. Docker Compose handles this automatically.
*   **Prometheus:** Time-series database for metrics. Docker Compose handles this automatically.
*   **Grafana:** Visualization platform for metrics. Docker Compose handles this automatically.
*   **Node Exporter:** Exposes system metrics for Prometheus. Docker Compose handles this automatically.
*   **Pre-trained Model:** The classification model weights file specified in `utils/config.py`.

### Python Packages:
Listed in `requirements.txt`. Key libraries include:
*   Python 3.8+
*   FastAPI, Uvicorn
*   PyTorch
*   Jinja2
*   python-multipart
*   passlib[bcrypt]
*   python-jose[cryptography]
*   pymongo
*   aiofiles (likely)
*   prometheus-client
*   python-dotenv (for loading `.env` file)

## Setup and Installation

### Option 1: Using Docker (Recommended)

This method runs the application, database, and monitoring stack in isolated containers.

1.  **Clone the Repository:**
    ```bash
    git clone <repository-url>
    cd BioSight/client-side # Navigate into the client-side directory
    ```

2.  **Prepare Environment Variables:**
    *   Copy the example environment file:
        ```bash
        cp .env.example .env
        ```
    *   Edit the `.env` file and **change the `SECRET_KEY`** to a strong, unique value.
    *   Verify `MONGODB_URI` (usually `mongodb://mongo:27017/` where `mongo` is the service name in `docker-compose.yml`).
    *   Ensure `DATABASE_NAME` is set.

3.  **Place Model Weights:**
    *   Download the required model weights file.
    *   Place it in `biosight/models/weights/`. Ensure the path matches `MODEL_SETTINGS['weights_file']` in `utils/config.py`.

4.  **Build and Run with Docker Compose:**
    ```bash
    docker-compose up --build -d
    ```
    *   `--build`: Forces Docker to rebuild images if the `Dockerfile` or source code has changed.
    *   `-d`: Runs the containers in detached mode.

5.  **Access Services:**
    *   **BioSight Application:** `http://localhost:8000` (or the port mapped for the `app` service).
    *   **Grafana:** `http://localhost:3000` (Default user/pass: admin/admin). The BioSight dashboard should be automatically provisioned.
    *   **Prometheus:** `http://localhost:9090`.

6.  **Stopping the Application Stack:**
    ```bash
    docker-compose down
    ```

### Option 2: Manual Local Setup (Without Docker - Monitoring Stack Not Included)

This method requires manual installation of dependencies and only runs the FastAPI application and MongoDB. Prometheus and Grafana need separate setup if desired.

1.  **Clone the Repository:** (As above)

2.  **Install MongoDB:** Install and run MongoDB locally. Note the connection URI (e.g., `mongodb://localhost:27017/`).

3.  **Create and Activate a Virtual Environment:**
    ```bash
    python3 -m venv venv
    source venv/bin/activate
    # On Windows use `venv\Scripts\activate`
    ```

4.  **Install Dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

5.  **Configure Environment Variables:**
    *   Create a `.env` file in the `client-side` directory.
    *   Add necessary configurations:
        ```env
        # .env
        MONGODB_URI="mongodb://localhost:27017/" # Adjust if needed
        DATABASE_NAME="biosight_db"
        SECRET_KEY="your-very-strong-secret-key-here" # Change this!
        ```

6.  **Place Model Weights:** (As described in the Docker setup)

7.  **Run the Application:**
    ```bash
    uvicorn biosight.app:app --reload --host 0.0.0.0 --port 8000
    ```

8.  **Access the Application:** `http://localhost:8000`.

## Usage

1.  **Register/Login:** Use the BioSight application at `http://localhost:8000`.
2.  **Upload/Manage Images:** Use the application interface as described previously.
3.  **Monitor Performance:** Access the Grafana dashboard at `http://localhost:3000` to view application and system metrics.

## Configuration

*   **Environment Variables (`.env`):** Primary configuration (database URI, secret key).
*   **Application Settings (`utils/config.py`):** Specific paths (`UPLOAD_FOLDER`, `ORGANIZED_FOLDER`), model parameters (`MODEL_SETTINGS`).
*   **Prometheus (`prometheus/prometheus.yml`):** Defines scrape targets (BioSight app, node-exporter).
*   **Grafana (`grafana/provisioning`):** Configures the Prometheus data source and automatically loads the dashboard from `grafana/dashboards/model.json`.

## Monitoring Details

*   **Prometheus:** Scrapes the `/metrics` endpoint exposed by the FastAPI application and the `/metrics` endpoint of the `node-exporter` service.
*   **Grafana:** Visualizes data queried from Prometheus. The default dashboard (`BioSight Application & System Monitoring`) includes panels for:
    *   System Metrics (CPU, Memory, Disk IO/Space, Network IO, File Descriptors) via Node Exporter.
    *   Application Metrics (Total Uploads, Predictions per Class) via FastAPI app.
    *   User Activity (Top Users by Uploads).