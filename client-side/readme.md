
# BioSight - Biological Image Classification and Organization

## Overview

BioSight is a web application built with FastAPI that allows users to upload biological images, automatically classify them using a pre-trained machine learning model (likely PyTorch-based), and organize the images into folders based on their predicted class. Users can register, log in, manage their uploaded images, correct classifications, and download the organized dataset.

## Features

*   **User Authentication:** Secure registration and login using JWT tokens stored in cookies.
*   **Image Upload:** Supports uploading multiple image files (`.png`, `.jpg`, `.jpeg`).
*   **Automatic Classification:** Classifies uploaded images into biological categories (e.g., Mammalia, Plantae, Amphibia, etc.).
*   **Image Organization:** Automatically moves classified images into corresponding folders.
*   **Results Visualization:** Displays classified images grouped by predicted class in a tabbed interface.
*   **Classification Correction:** Allows users to manually change the assigned class of an image.
*   **Image Deletion:** Users can delete uploaded images.
*   **Download:** Option to download all organized images as a single Zip archive.
*   **API Monitoring:** Includes a `/metrics` endpoint for Prometheus monitoring (prediction counts, latency, etc.).
*   **Docker Support:** Includes configuration for running the application and its dependencies using Docker Compose.

## File Structure

```
/BioSight/
├── client-side/
│   └── biosight/
│       ├── __init__.py
│       ├── app.py             # Main FastAPI application logic and routes
│       ├── static/            # Static files served directly
│       │   ├── css/
│       │   │   └── style.css
│       │   └── js/
│       │       └── auth.js    # (Assumed) JavaScript for frontend interactions
│       ├── templates/         # Jinja2 HTML templates
│       │   ├── index.html
│       │   ├── login.html
│       │   ├── register.html
│       │   └── result.html
│       ├── routes/            # API route modules
│       │   ├── __init__.py
│       │   ├── auth.py
│       │   └── user.py
│       ├── utils/             # Utility modules
│       │   ├── __init__.py
│       │   ├── config.py
│       │   ├── constants.py
│       │   ├── database.py
│       │   ├── file_operations.py
│       │   ├── image_processor.py
│       │   ├── model_loader.py
│       │   ├── monitoring.py
│       │   └── security.py
│       └── models/            # Directory for ML model files
│           └── weights/
│               └── model_weights.pth # Example location for model weights
├── Dockerfile             # Instructions to build the FastAPI application image
├── docker-compose.yml     # Defines services (app, db) for Docker Compose
├── requirements.txt       # Python package dependencies
├── README.md              # This file
└── .env.example           # Example environment variables file
```
*(Note: `Dockerfile`, `docker-compose.yml`, and `.env.example` might need to be created if they don't exist)*

## Dependencies

### External Services:
*   **Docker:** Required for containerization. Install Docker Desktop (Windows/Mac) or Docker Engine (Linux).
*   **Docker Compose:** Usually included with Docker Desktop, or install separately on Linux.
*   **MongoDB:** A running MongoDB instance is required. Docker Compose handles this automatically when using the provided setup.
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

This method runs the application and its MongoDB dependency in isolated containers.

1.  **Clone the Repository:**
    ```bash
    git clone <repository-url>
    cd BioSight
    ```

2.  **Prepare Environment Variables:**
    *   Copy the example environment file:
        ```bash
        cp .env.example .env
        ```
    *   Edit the `.env` file and **change the `SECRET_KEY`** to a strong, unique value.
    *   Adjust `MONGODB_URI` if your Docker setup requires a different hostname for the MongoDB service (usually `mongodb://mongo:27017/` where `mongo` is the service name in `docker-compose.yml`).
    *   Ensure `DATABASE_NAME` is set.

3.  **Place Model Weights:**
    *   Download the required model weights file.
    *   Place it in the location expected by the application, typically within `client-side/biosight/models/weights/`. Ensure the path matches `MODEL_SETTINGS['weights_file']` in `utils/config.py`. This directory needs to be accessible within the Docker container (check volume mounts in `docker-compose.yml`).

4.  **Build and Run with Docker Compose:**
    ```bash
    docker-compose up --build -d
    ```
    *   `--build`: Forces Docker to rebuild the application image if the `Dockerfile` or source code has changed.
    *   `-d`: Runs the containers in detached mode (in the background).

5.  **Access the Application:** Open your web browser and go to `http://localhost:8000` (or the port mapped in `docker-compose.yml`).

6.  **Stopping the Application:**
    ```bash
    docker-compose down
    ```

### Option 2: Manual Local Setup (Without Docker)

1.  **Clone the Repository:** (As above)

2.  **Install MongoDB:** Install and run MongoDB locally following the official MongoDB documentation for your operating system. Note the connection URI (e.g., `mongodb://localhost:27017/`).

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
    *   Create a `.env` file in the root directory (`/BioSight/`).
    *   Add necessary configurations:
        ```env
        # .env
        MONGODB_URI="mongodb://localhost:27017/" # Adjust if needed
        DATABASE_NAME="biosight_db"
        SECRET_KEY="your-very-strong-secret-key-here" # Change this!
        ```
    *   Ensure the application code (e.g., `utils/database.py`, `utils/security.py`) is configured to load these variables (using `python-dotenv`).

6.  **Place Model Weights:** (As described in the Docker setup)

7.  **Run the Application:**
    *   Navigate to the application directory:
        ```bash
        cd client-side
        ```
    *   Start the FastAPI server using Uvicorn:
        ```bash
        uvicorn biosight.app:app --reload --host 0.0.0.0 --port 8000
        ```

8.  **Access the Application:** Open your web browser and go to `http://localhost:8000`.

## Usage

1.  **Register:** Access the application URL and use the registration form to create an account.
2.  **Login:** Log in with your registered email and password.
3.  **Upload:** On the main page (`/`), select image files and click "Upload and Classify".
4.  **View Results:** You'll be redirected to the results page (`/results` - path might vary) where images are grouped by predicted class.
5.  **Manage Images:**
    *   Switch between class tabs.
    *   Change an image's classification using the dropdown menu.
    *   Delete an image using the '×' button.
6.  **Download:** Click "Download All as Zip" to get an archive of your organized images.
7.  **Logout:** Click the "Logout" button.

## Configuration

*   **Environment Variables:** Primary configuration (database URI, secret key) should be managed via the `.env` file.
*   **Application Settings:** Specific paths (`UPLOAD_FOLDER`, `ORGANIZED_FOLDER`) and model parameters (`MODEL_SETTINGS`) are defined in `client-side/biosight/utils/config.py`. Ensure these paths exist and have correct permissions, especially when running locally or defining Docker volumes.