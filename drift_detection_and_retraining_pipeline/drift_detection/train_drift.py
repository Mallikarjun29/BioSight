import os
import torch
import numpy as np
from alibi_detect.cd import MMDDrift
from alibi_detect.utils.pytorch import get_device
from prepare_data import DataPreparation
from model import load_model

class FeatureExtractor(torch.nn.Module):
    def __init__(self, base_model, device):
        super().__init__()
        self.feature_extractor = torch.nn.Sequential(*(list(base_model.children())[:-1]))
        self.device = device
    
    def forward(self, x):
        features = self.feature_extractor(x.to(self.device))
        return features.view(features.size(0), -1)

def collect_features(loader, feature_extractor, device, max_samples=1000):
    """Collect features from data loader for drift detection"""
    features_list = []
    n_samples = 0
    
    with torch.no_grad():
        for imgs, _ in loader:
            if n_samples >= max_samples:
                break
            imgs = imgs.to(device)
            features = feature_extractor(imgs).cpu().numpy()
            features_list.append(features)
            n_samples += features.shape[0]
            
    return np.concatenate(features_list, axis=0)[:max_samples]

def save_drift_detector(drift_detector, save_dir):
    """Save the drift detector object"""
    import joblib
    
    os.makedirs(save_dir, exist_ok=True)
    # Save the entire detector object
    joblib.dump(drift_detector, os.path.join(save_dir, 'drift_detector.joblib'))
    print("Drift detector saved successfully")

if __name__ == "__main__":
    # Load pretrained model
    model = load_model(
        model_type="resnet50",
        weights_path="DEFAULT",
        num_classes=10
    )
    
    device = get_device()
    print(f"Using device: {device}")
    model = model.to(device)
    import os
    from pathlib import Path

    # Calculate the correct path relative to this script's location
    script_location = Path(__file__).parent
    data_directory = script_location.parent.parent / "inaturalist_12K" # Go up two levels to BioSight, then into dataset

    # Create feature extractor
    feature_extractor = FeatureExtractor(model, device)
    feature_extractor.eval()
    
    # Set up data loaders
    data_dir = "../inaturalist_12K"
    batch_size = 64
    data_prep = DataPreparation(str(data_directory), batch_size=batch_size)
    train_loader, val_loader, _ = data_prep.get_data_loaders()
    
    # Collect reference features
    print("Collecting reference features...")
    reference_features = collect_features(train_loader, feature_extractor, device)
    
    # Initialize drift detector with correct kernel configuration
    drift_detector = MMDDrift(
        x_ref=reference_features,
        backend='pytorch',
        p_val=0.01,  # This p-value threshold is stored internally
        preprocess_fn=None,
        n_permutations=100
    )
    
    print("Drift detector initialized with reference distribution")

    # Save models and configuration
    save_dir = "drift_models"
    os.makedirs(save_dir, exist_ok=True)
    
    # Save feature extractor
    torch.save({
        'model_state_dict': feature_extractor.state_dict(),
        'device': str(device),
    }, os.path.join(save_dir, 'feature_extractor.pth'))
    
    # Save drift detector using the simplified method
    save_drift_detector(drift_detector, save_dir)
    print(f"Models saved in {save_dir}/")

    # Test drift detection
    print("\nTesting drift detection...")
    test_features = collect_features(val_loader, feature_extractor, device, max_samples=200)
    predictions = drift_detector.predict(test_features)
    
    print(f"Drift test p-value: {predictions['data']['p_val']:.4f}")
    print(f"Drift detected: {predictions['data']['is_drift']}")
