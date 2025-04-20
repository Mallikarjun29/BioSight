"""FastAPI web application for image classification and organization."""
import sys
from datetime import datetime, timezone
from pathlib import Path
import time

from fastapi import FastAPI, File, UploadFile, HTTPException, Request, Depends
from fastapi.responses import HTMLResponse, FileResponse, JSONResponse, RedirectResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
import torch


# Import configurations and utilities
try:
    from biosight.utils.config import UPLOAD_FOLDER, MODEL_SETTINGS
    from biosight.utils.constants import STATUS_MESSAGES
    from biosight.utils.model_loader import load_model as load_model_from_utils
    from biosight.utils.database import db
    from biosight.utils.file_operations import (
        allowed_file, clear_organized_folder, save_upload_file,
        organize_file, create_zip_archive
    )
    from biosight.utils.image_processor import ImageProcessor
    from biosight.utils.monitoring import PREDICTION_COUNTER, PREDICTION_LATENCY, UPLOAD_COUNTER, get_metrics
    from biosight.routes.auth import router as auth_router
    from biosight.utils.security import get_current_user  # Import from utils.security instead of routes.auth
    from biosight.routes.user import User  # Fixed import path for User model
except ImportError as e:
    print(f"Error: Could not import from biosight: {e}")
    sys.exit(1)

# Base directory and static paths configuration
BASE_DIR = Path(__file__).parent
STATIC_DIR = BASE_DIR / "static"
TEMPLATES_DIR = BASE_DIR / "templates"

# Create directories if they don't exist
STATIC_DIR.mkdir(parents=True, exist_ok=True)

app = FastAPI()

# Configure static files and templates with absolute paths
app.mount("/uploads", StaticFiles(directory=UPLOAD_FOLDER), name="uploads")
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
templates = Jinja2Templates(directory=TEMPLATES_DIR)

# Include authentication routes
app.include_router(auth_router)

# Initialize database connection
if not db.connect():
    print("Failed to connect to database. Exiting.")
    sys.exit(1)

# Initialize model and image processor
try:
    print("Initializing model...")
    model = load_model_from_utils(
        model_type=MODEL_SETTINGS['model_type'],
        weights_path=MODEL_SETTINGS['weights_file'],
        num_classes=MODEL_SETTINGS['num_classes'],
        dropout_rate=MODEL_SETTINGS['dropout_rate']
    )
    
    if model:
        print("Model loaded successfully")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
        image_processor = ImageProcessor(model)
    else:
        print("Model loading failed")
        sys.exit(1)
except Exception as e:
    print(f"Error loading model: {str(e)}")
    sys.exit(1)

@app.get("/login", response_class=HTMLResponse)
async def login_page(request: Request):
    return templates.TemplateResponse("login.html", {"request": request})

@app.get("/register", response_class=HTMLResponse)
async def register_page(request: Request):
    return templates.TemplateResponse("register.html", {"request": request})

@app.get("/", response_class=HTMLResponse)
async def home(request: Request, current_user: User = Depends(get_current_user)):
    """Home page - protected by authentication"""
    return templates.TemplateResponse("index.html", {
        "request": request,
        "user": current_user  # Pass authenticated user to template
    })

@app.get("/metrics")
async def metrics():
    """Endpoint for Prometheus metrics."""
    return get_metrics()

@app.post("/upload/")
async def upload_files(
    request: Request, 
    files: list[UploadFile] = File(...),
    current_user: User = Depends(get_current_user)  # Add authentication requirement
):
    """Handle multiple file uploads, classify them, and store results."""
    if not files:
        raise HTTPException(
            status_code=400,
            detail=STATUS_MESSAGES['NO_FILES']
        )

    # Clear organized folder before processing new files
    clear_organized_folder()
    results = []

    for file in files:
        if not file.filename:
            continue

        if not allowed_file(file.filename):
            raise HTTPException(status_code=400, detail=f"Invalid file type: {file.filename}")

        UPLOAD_COUNTER.inc()  # Increment upload counter

        # Save uploaded file
        upload_file_path, random_name = await save_upload_file(file)

        # Time and record the prediction
        start_time = time.time()
        predicted_class = image_processor.predict(str(upload_file_path))
        PREDICTION_LATENCY.observe(time.time() - start_time)
        PREDICTION_COUNTER.labels(predicted_class).inc()

        # Organize file
        organized_file_path = organize_file(upload_file_path, random_name, predicted_class)

        # Save metadata
        metadata = {
            "original_filename": file.filename,
            "random_name": random_name,
            "upload_path": str(upload_file_path),
            "organized_path": str(organized_file_path),
            "predicted_class": predicted_class,
            "timestamp": datetime.now(timezone.utc),
            "drift_detected": False,
            "used_in_training": False
        }

        if not db.save_metadata(metadata):
            if organized_file_path.exists():
                organized_file_path.unlink()
            raise HTTPException(
                status_code=500,
                detail=STATUS_MESSAGES['DB_ERROR'].format(file.filename)
            )

        results.append({
            "filename": file.filename,
            "predicted_class": predicted_class,
            "saved_path": str(organized_file_path)
        })

    # Create zip archive
    zip_file_path = create_zip_archive()

    return templates.TemplateResponse(
        "result.html",
        {
            "request": request,
            "results": results,
            "zip_file_path": zip_file_path,
        }
    )

@app.get("/download-zip/")
async def download_zip(current_user: User = Depends(get_current_user)):
    """Download the zip file of organized images."""
    zip_file_path = create_zip_archive()
    if not zip_file_path or not Path(zip_file_path).exists():
        raise HTTPException(status_code=404, detail="Zip file not found and could not be created.")
    return FileResponse(zip_file_path, media_type="application/zip")

@app.get("/health")
async def health_check():
    """Basic health check endpoint."""
    mongo_status = db.check_health()
    model_status = "ok" if model else "error: model not loaded"
    status_code = 200 if mongo_status == "ok" and model_status == "ok" else 503

    return JSONResponse(
        status_code=status_code,
        content={
            "status": "ok" if status_code == 200 else "error",
            "mongodb": mongo_status,
            "model": model_status
        }
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)