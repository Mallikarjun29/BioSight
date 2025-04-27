"""FastAPI web application for image classification and organization."""
import sys
from datetime import datetime, timezone
from pathlib import Path
import time
import logging
from typing import Optional

from fastapi import FastAPI, File, UploadFile, HTTPException, Request, Depends, status
from fastapi.responses import HTMLResponse, FileResponse, JSONResponse, RedirectResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint # Import middleware components
from starlette.responses import Response as StarletteResponse # Import Response for middleware
import torch
import shutil

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# Import configurations and utilities
try:
    from biosight.utils.config import UPLOAD_FOLDER, MODEL_SETTINGS, ORGANIZED_FOLDER
    from biosight.utils.constants import STATUS_MESSAGES
    from biosight.utils.model_loader import load_model as load_model_from_utils
    from biosight.utils.database import Database, db  # Updated import
    from biosight.utils.file_operations import (
        allowed_file, clear_organized_folder, save_upload_file,
        organize_file, create_zip_archive
    )
    from biosight.utils.image_processor import ImageProcessor
    # Import the new HTTP counter
    from biosight.utils.monitoring import PREDICTION_COUNTER, PREDICTION_LATENCY, UPLOAD_COUNTER, HTTP_REQUESTS_TOTAL, get_metrics 
    from biosight.routes.auth import router as auth_router
    # Import both user dependency functions
    from biosight.utils.security import get_current_user, get_current_user_optional 
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

# --- Add Request Counting Middleware ---
class MetricsMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint) -> StarletteResponse:
        # Exclude metrics endpoint itself from being counted
        if request.url.path != "/metrics":
            start_time = time.time()
            response = await call_next(request)
            process_time = time.time() - start_time
            
            # Use route template if available for better grouping, otherwise use raw path
            route_template = request.scope.get("route").path if request.scope.get("route") else request.url.path
            
            HTTP_REQUESTS_TOTAL.labels(method=request.method, path=route_template).inc()
            # You could add another histogram here for request latency if desired
            # HTTP_REQUEST_LATENCY.labels(method=request.method, path=route_template).observe(process_time)
            
            return response
        else:
            # Don't count metrics requests
            return await call_next(request)

app.add_middleware(MetricsMiddleware)
# --- End Middleware ---

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configure static files and templates with absolute paths
app.mount("/uploads", StaticFiles(directory=UPLOAD_FOLDER), name="uploads")
app.mount("/organized", StaticFiles(directory=ORGANIZED_FOLDER), name="organized")
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

# --- Route Definitions ---

@app.get("/login", response_class=HTMLResponse)
async def login_page(request: Request):
    return templates.TemplateResponse("login.html", {"request": request})

@app.get("/register", response_class=HTMLResponse)
async def register_page(request: Request):
    return templates.TemplateResponse("register.html", {"request": request})

@app.get("/", response_class=HTMLResponse)
async def home(request: Request, current_user: Optional[User] = Depends(get_current_user_optional)):
    """Home page - redirects to login if not authenticated."""
    if current_user is None:
        # User not authenticated, redirect to login page
        return RedirectResponse(url="/login", status_code=status.HTTP_307_TEMPORARY_REDIRECT)
        
    # User is authenticated, render the index page
    return templates.TemplateResponse("index.html", {
        "request": request,
        "user": current_user  # Pass authenticated user to template
    })

@app.get("/metrics")
async def metrics():
    """Endpoint for Prometheus metrics."""
    return get_metrics()

# Ensure other protected routes still use the strict dependency
@app.post("/upload/")
async def upload_files(
    request: Request,
    files: list[UploadFile] = File(...),
    current_user: User = Depends(get_current_user)
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
    user_email = current_user.email # Get user email

    for file in files:
        if not file.filename:
            continue

        if not allowed_file(file.filename):
            raise HTTPException(status_code=400, detail=f"Invalid file type: {file.filename}")

        UPLOAD_COUNTER.labels(user_email=user_email).inc() # Add user_email label

        # Save uploaded file
        upload_file_path, random_name = await save_upload_file(file)

        # Time and record the prediction
        start_time = time.time()
        predicted_class = image_processor.predict(str(upload_file_path))
        PREDICTION_LATENCY.labels(user_email=user_email).observe(time.time() - start_time) # Add user_email label
        PREDICTION_COUNTER.labels(class_name=predicted_class, user_email=user_email).inc() # Add user_email label

        # Organize file
        organized_file_path = organize_file(upload_file_path, random_name, predicted_class)

        # Create a web-accessible path for the template
        relative_path = str(organized_file_path).replace(str(BASE_DIR), '')
        if relative_path.startswith('/'):
            relative_path = relative_path[1:]

        # Save metadata
        metadata = {
            "original_filename": file.filename,
            "random_name": random_name,
            "upload_path": str(upload_file_path),
            "organized_path": str(organized_file_path),
            "predicted_class": predicted_class,
            "original_predicted_class": predicted_class,  # Store original prediction
            "is_updated": False,  # Track if classification has been updated
            "timestamp": datetime.now(timezone.utc),
            "drift_detected": False,
            "used_in_training": False,
            "user_email": user_email # Already storing it here
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
            "saved_path": relative_path,
            "is_updated": metadata["is_updated"]
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

@app.delete("/delete-image/{predicted_class}/{filename}")
async def delete_image(predicted_class: str, filename: str, current_user: User = Depends(get_current_user)):
    """Delete an image from the organized folder and its metadata from the database."""
    try:
        # Construct the file path using ORGANIZED_FOLDER from config
        file_path = ORGANIZED_FOLDER / predicted_class / filename
        
        if not file_path.exists():
            raise HTTPException(status_code=404, detail="Image not found")
        
        # Delete the file
        file_path.unlink()
        
        # Delete metadata from database using the images collection
        try:
            result = db.collection.delete_one({
                "random_name": filename,
                "predicted_class": predicted_class
            })
            if result.deleted_count == 0:
                logger.warning(f"No metadata found for image {filename}")
        except Exception as db_error:
            logger.error(f"Database error while deleting metadata: {str(db_error)}")
        
        return {"message": "Image deleted successfully"}
    except Exception as e:
        logger.error(f"Error deleting image {filename}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error deleting image: {str(e)}")

@app.put("/update-class/{old_class}/{filename}")
async def update_image_class(
    old_class: str,
    filename: str,
    new_class: str,
    current_user: User = Depends(get_current_user)
):
    """Update the class of an image and move it to the new class folder."""
    try:
        # Get metadata first to determine current location and original class
        metadata = db.collection.find_one({"random_name": filename})
        if not metadata:
            logger.error(f"Metadata not found for image {filename}")
            raise HTTPException(status_code=404, detail="Image metadata not found")
            
        # Get the current class and original class from metadata
        current_class = metadata.get("predicted_class", old_class)
        original_predicted_class = metadata.get("original_predicted_class")
        
        # Log the class information for debugging
        logger.info(f"Moving file {filename} from {current_class} to {new_class}. Original class was {original_predicted_class}")
        
        # Determine if this is a reversion to original class
        is_reverting = (original_predicted_class == new_class)
        logger.info(f"Is reverting: {is_reverting}")
        
        # Find the file in its current location
        current_path = ORGANIZED_FOLDER / current_class / filename
        
        # If not found in the expected folder, search in all class folders
        if not current_path.exists():
            # Get all subdirectories in the organized folder
            class_folders = [d for d in ORGANIZED_FOLDER.iterdir() if d.is_dir()]
            file_found = False
            
            # Search for the file in all class folders
            for class_folder in class_folders:
                potential_path = class_folder / filename
                if potential_path.exists():
                    current_path = potential_path
                    current_class = class_folder.name
                    file_found = True
                    logger.info(f"Found file in {current_class} folder instead of expected {old_class}")
                    break
            
            if not file_found:
                logger.error(f"Image not found in any class folder: {filename}")
                raise HTTPException(status_code=404, detail="Image not found in any class folder")

        # Prepare new path
        new_path = ORGANIZED_FOLDER / new_class / filename
        
        # Create new class directory if it doesn't exist
        (ORGANIZED_FOLDER / new_class).mkdir(parents=True, exist_ok=True)
        
        try:
            # Move the file
            shutil.move(str(current_path), str(new_path))
            logger.info(f"Successfully moved file from {current_path} to {new_path}")
        except Exception as move_error:
            logger.error(f"Error moving file from {current_path} to {new_path}: {str(move_error)}")
            raise HTTPException(status_code=500, detail=f"Error moving file: {str(move_error)}")
        
        try:
            # Always update the predicted class in the database
            update_data = {
                "predicted_class": new_class,
                "is_updated": not is_reverting,  # Set to False if reverting to original class
                "last_modified": datetime.now(timezone.utc)
            }
            
            # Update metadata in database
            result = db.collection.update_one(
                {"random_name": filename},
                {"$set": update_data}
            )
            
            if result.modified_count == 0:
                logger.warning(f"No metadata updated for image {filename}")
                
        except Exception as db_error:
            logger.error(f"Database error while updating metadata: {str(db_error)}")
            # Try to move the file back to its original location
            try:
                shutil.move(str(new_path), str(current_path))
            except Exception as rollback_error:
                logger.error(f"Failed to rollback file move after database error: {str(rollback_error)}")
            raise HTTPException(status_code=500, detail=f"Database error: {str(db_error)}")
        
        # Return the successful response with all needed data
        return {
            "message": "Class updated successfully",
            "old_class": current_class,
            "new_class": new_class,
            "is_updated": not is_reverting,
            "original_class": original_predicted_class
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error updating class for image {filename}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error updating class: {str(e)}")

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