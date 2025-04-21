"""Configuration settings for the BioSight application."""

import os
from pathlib import Path

# Base Paths
BASE_DIR = Path(__file__).parent.parent
MODELS_DIR = BASE_DIR / "models"
UPLOAD_FOLDER = BASE_DIR / "uploads"
ORGANIZED_FOLDER = BASE_DIR / "organized_images"

# Ensure directories exist
MODELS_DIR.mkdir(parents=True, exist_ok=True)
UPLOAD_FOLDER.mkdir(parents=True, exist_ok=True)
ORGANIZED_FOLDER.mkdir(parents=True, exist_ok=True)

# File Settings
ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg"}
ZIP_FILENAME_BASE = "organized_images"

# MongoDB Settings
MONGO_URI = os.getenv('MONGO_URI', "mongodb://localhost:27017/")
DB_NAME = os.getenv('MONGO_DB', "image_organizer")
COLLECTION_NAME = os.getenv('MONGO_COLLECTION', "image_metadata")

# Model Settings
MODEL_SETTINGS = {
    'model_type': 'resnet50',
    'weights_file': str(MODELS_DIR / 'best_model_resnet.pth'),  # Using absolute path
    'num_classes': 10,
    'dropout_rate': 0.5,
    'confidence_threshold': 0.5
}

# Image Preprocessing
IMAGE_TRANSFORMS = {
    'resize': 224,
    'crop_size': 224,
    'mean': [0.485, 0.456, 0.406],
    'std': [0.229, 0.224, 0.225]
}