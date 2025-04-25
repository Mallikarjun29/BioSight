import os
import torch
import joblib
from alibi_detect.cd import MMDDrift # Ensure MMDDrift is imported
from alibi_detect.utils.pytorch import get_device # Ensure get_device is imported
from train_drift import FeatureExtractor # Ensure FeatureExtractor is imported
from prepare_data import DataPreparation
from model import load_model
import numpy as np # Make sure numpy is imported

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
         print(f"Warning: Feature extractor checkpoint not found at {fe_path}")
    
    feature_extractor.to(device) # Ensure model is on the correct device after loading
    feature_extractor.eval()
    
    # Load drift detector object directly
    detector_path = os.path.join(model_dir, 'drift_detector.joblib')
    if os.path.exists(detector_path):
        drift_detector = joblib.load(detector_path)
    else:
        raise RuntimeError(f"Drift detector not found at {detector_path}")
    
    # Ensure the loaded detector uses the correct device if applicable (depends on backend)
    if hasattr(drift_detector, 'to'):
         drift_detector.to(device)
    elif hasattr(drift_detector, 'device'):
         drift_detector.device = device

    return feature_extractor, drift_detector, device

if __name__ == "__main__":
    # Test the loader
    feature_extractor, drift_detector, device = load_drift_models()
    print("Models loaded successfully")
    
    # Load some test data
    data_dir = "inaturalist_12K"
    batch_size = 4  # Smaller batch size for testing
    data_prep = DataPreparation(data_dir, batch_size=batch_size, val_split=0.2)
    _, _, test_loader = data_prep.get_data_loaders()
    
    print("\nTesting drift detection on one batch...")
    # Test drift detection
    with torch.inference_mode():
        for imgs, _ in test_loader:
            imgs = imgs.to(device)
            # Extract features and convert to numpy for alibi-detect
            features_np = feature_extractor(imgs).cpu().numpy()
            print(f"Features shape: {features_np.shape}")
            
            # Use predict method
            predictions = drift_detector.predict(features_np)
            
            # Extract results from the prediction dictionary
            p_val = float(predictions['data']['p_val'])
            is_drift = bool(predictions['data']['is_drift'])
            threshold = float(predictions['data']['threshold']) # Get the threshold used by the detector
            
            print(f"P-value: {p_val:.6f}")
            print(f"Threshold: {threshold:.6f}")
            
            if is_drift:
                print(f"WARNING: Drift detected (p-value {p_val:.6f} < {threshold:.6f})")
            else:
                print(f"No significant drift detected (p-value {p_val:.6f} >= {threshold:.6f})")
            
            # Only process one batch for testing
            break