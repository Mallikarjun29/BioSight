import os
import torch
import torchdrift
from prepare_data import DataPreparation
from model import load_model
from drift import FeatureExtractor

def load_drift_models(model_dir="drift_models"):
    # Load base model
    model_type = "resnet50"
    weights_path = "best_model_resnet.pth"
    num_classes = 10
    dropout_rate = 0.5
    
    model = load_model(model_type, weights_path, num_classes=num_classes, dropout_rate=dropout_rate)
    if model is None:
        raise RuntimeError("Failed to load base model.")
    
    # Load feature extractor
    fe_checkpoint = torch.load(os.path.join(model_dir, 'feature_extractor.pth'))
    device = fe_checkpoint['device']
    model = model.to(device)
    
    feature_extractor = FeatureExtractor(model, device)
    feature_extractor.load_state_dict(fe_checkpoint['model_state_dict'])
    feature_extractor.eval()
    
    # Load drift detector configuration
    detector_config = torch.load(os.path.join(model_dir, 'drift_detector.pth'), weights_only=True)
    
    # Create new detector with loaded configuration
    drift_detector = torchdrift.detectors.KernelMMDDriftDetector()
    if detector_config['kernel_sigma'] is not None:
        drift_detector.kernel.sigma = detector_config['kernel_sigma']
    if detector_config['base_outputs'] is not None:
        drift_detector.base_outputs = detector_config['base_outputs']
    
    return feature_extractor, drift_detector, device

if __name__ == "__main__":
    # Load the models
    feature_extractor, drift_detector, device = load_drift_models()
    print("Models loaded successfully")
    
    # Load some test data
    data_dir = "inaturalist_12K"
    batch_size = 32
    data_prep = DataPreparation(data_dir, batch_size=batch_size, val_split=0.2)
    _, _, test_loader = data_prep.get_data_loaders()
    
    # Test drift detection
    with torch.no_grad():
        for imgs, _ in test_loader:
            imgs = imgs.to(device)
            features = feature_extractor(imgs)
            print(f"Features shape: {features.shape}")
            
            score = drift_detector(features)
            p_val = drift_detector.compute_p_value(features)
            
            print(f"Drift score: {score.item():.4f} (lower is better)")
            print(f"P-value: {p_val:.4f} (higher is better)")
            
            DRIFT_THRESHOLD = 0.01
            if p_val < DRIFT_THRESHOLD:
                print(f"WARNING: Drift detected (p-value {p_val:.4f} < {DRIFT_THRESHOLD})")
            else:
                print(f"No significant drift detected (p-value {p_val:.4f} >= {DRIFT_THRESHOLD})")
            break