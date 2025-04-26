import os
import torch
import joblib  # Use joblib consistently
import logging
from alibi_detect.cd import MMDDrift
from alibi_detect.utils.pytorch import get_device
from train_drift import FeatureExtractor
from prepare_data import DataPreparation
from model import load_model
import numpy as np

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

def load_drift_models(model_dir="drift_models"):
    """Load feature extractor and drift detector models"""
    
    # Load pretrained model
    model = load_model(
        model_type="resnet50",
        weights_path="DEFAULT",
        num_classes=10
    )
    if model is None:
        raise RuntimeError("Failed to load pretrained model.")
    
    device = get_device()
    model = model.to(device)
    model.eval()
    
    # Create and load feature extractor
    feature_extractor = FeatureExtractor(model, device)
    
    fe_path = os.path.join(model_dir, 'feature_extractor.pth')
    if os.path.exists(fe_path):
        # Load state dict based on device
        map_location = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        fe_checkpoint = torch.load(fe_path, map_location=map_location)
        feature_extractor.load_state_dict(fe_checkpoint['model_state_dict'])
    else:
         logger.warning(f"Feature extractor checkpoint not found at {fe_path}")
    
    feature_extractor.to(device) # Ensure model is on the correct device after loading
    feature_extractor.eval()
    
    # Load drift detector object directly
    detector_path = os.path.join(model_dir, 'drift_detector.joblib')
    if os.path.exists(detector_path):
        try:
            # Use joblib.load instead of pickle.load
            drift_detector = joblib.load(detector_path)
            logger.info(f"Loaded drift detector from {detector_path}")
        except Exception as e:
            logger.error(f"Error loading drift detector: {e}")
            raise RuntimeError(f"Failed to load drift detector: {e}")
    else:
        raise RuntimeError(f"Drift detector not found at {detector_path}")
    
    # Ensure the loaded detector uses the correct device if applicable (depends on backend)
    if hasattr(drift_detector, 'to'):
         drift_detector.to(device)
    elif hasattr(drift_detector, 'device'):
         drift_detector.device = device

    return feature_extractor, drift_detector, device