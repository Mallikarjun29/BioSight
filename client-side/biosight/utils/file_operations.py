"""File operation utilities."""
import os
import shutil
import logging
from pathlib import Path
import uuid
from fastapi import UploadFile, HTTPException
from .config import UPLOAD_FOLDER, ORGANIZED_FOLDER, ALLOWED_EXTENSIONS, ZIP_FILENAME_BASE
from .constants import STATUS_MESSAGES

logger = logging.getLogger(__name__)

def allowed_file(filename: str) -> bool:
    """Check if the file extension is allowed."""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def clear_organized_folder():
    """Clear all contents of the organized_images folder."""
    try:
        if ORGANIZED_FOLDER.exists():
            for f in ORGANIZED_FOLDER.glob("**/*"):
                if f.is_file():
                    try:
                        f.unlink(missing_ok=True)
                    except Exception as e:
                        logger.warning(f"Could not remove file {f}: {e}")
            # Remove empty directories
            for d in sorted(ORGANIZED_FOLDER.glob("**/*"), reverse=True):
                if d.is_dir():
                    try:
                        d.rmdir()
                    except Exception as e:
                        logger.warning(f"Could not remove directory {d}: {e}")
    except Exception as e:
        logger.error(f"Error while cleaning organized folder: {e}")

async def save_upload_file(file: UploadFile) -> tuple[Path, str]:
    """
    Save an uploaded file to the uploads directory.
    
    Returns:
        tuple: (file_path, random_name)
    """
    if not file.filename:
        raise HTTPException(status_code=400, detail="No filename provided")

    random_name = str(uuid.uuid4().hex[:16]) + Path(file.filename).suffix
    upload_file_path = UPLOAD_FOLDER / random_name

    try:
        with open(upload_file_path, "wb") as f:
            content = await file.read()
            f.write(content)
        return upload_file_path, random_name
    except Exception as e:
        if upload_file_path.exists():
            upload_file_path.unlink()
        raise HTTPException(
            status_code=500,
            detail=STATUS_MESSAGES['FILE_SAVE_ERROR'].format(str(e))
        )

def organize_file(upload_file_path: Path, random_name: str, predicted_class: str) -> Path:
    """
    Copy a file from uploads to its organized location based on predicted class.
    
    Returns:
        Path: Path to the organized file
    """
    class_folder = ORGANIZED_FOLDER / predicted_class
    class_folder.mkdir(parents=True, exist_ok=True)
    organized_file_path = class_folder / random_name

    try:
        shutil.copy2(str(upload_file_path), str(organized_file_path))
        return organized_file_path
    except Exception as e:
        if organized_file_path.exists():
            organized_file_path.unlink()
        raise HTTPException(
            status_code=500,
            detail=STATUS_MESSAGES['MOVE_ERROR'].format(str(e))
        )

def create_zip_archive() -> str:
    """
    Create a zip archive of the organized folder.
    
    Returns:
        str: Path to the created zip file or None if creation fails
    """
    try:
        zip_file_path = f"{ZIP_FILENAME_BASE}.zip"
        if Path(zip_file_path).exists():
            Path(zip_file_path).unlink()
        shutil.make_archive(ZIP_FILENAME_BASE, "zip", ORGANIZED_FOLDER)
        return zip_file_path
    except Exception as e:
        logger.warning(f"Could not create zip file: {e}")
        return None

# Create necessary directories
UPLOAD_FOLDER.mkdir(parents=True, exist_ok=True)
ORGANIZED_FOLDER.mkdir(parents=True, exist_ok=True)