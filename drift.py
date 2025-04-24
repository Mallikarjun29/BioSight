import os
import torch
import torchdrift
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

if __name__ == "__main__":
    data_dir = "inaturalist_12K"
    batch_size = 32
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Prepare data loaders
    data_prep = DataPreparation(data_dir, batch_size=batch_size, val_split=0.2)
    train_loader, val_loader, test_loader = data_prep.get_data_loaders()

    # Load your model (change model_type/weights as needed)
    model_type = "resnet50"
    weights_path = "best_model_resnet.pth"  # Or path to your .pth file
    num_classes = 10           # Set as needed
    dropout_rate = 0.5        # Set as needed

    model = load_model(model_type, weights_path, num_classes=num_classes, dropout_rate=dropout_rate)
    if model is None:
        raise RuntimeError("Failed to load model.")

    model = model.to(device)
    model.eval()

    # Create a proper feature extractor module
    feature_extractor = FeatureExtractor(model, device)
    feature_extractor.eval()

    # Create a drift detector
    drift_detector = torchdrift.detectors.KernelMMDDriftDetector()

    # Fit reference distribution on training data
    torchdrift.utils.fit(test_loader, feature_extractor, drift_detector, device=device)
    print("Drift detector fitted on training data.")

    # Example: check drift on a batch from validation set
    with torch.no_grad():
        for imgs, _ in test_loader:
            imgs = imgs.to(device)
            features = feature_extractor(imgs)
            print(f"Features shape: {features.shape}")
            
            score = drift_detector(features)
            p_val = drift_detector.compute_p_value(features)
            print(f"Drift score: {score.item():.4f}, p-value: {p_val:.4f}")
            if p_val < 0.01:
                print("Drift detected in this batch!")
            else:
                print("No drift detected in this batch.")
            break

    # Save the models
    save_dir = "drift_models"
    os.makedirs(save_dir, exist_ok=True)
    
    # Save feature extractor
    torch.save({
        'model_state_dict': feature_extractor.state_dict(),
        'device': device,
    }, os.path.join(save_dir, 'feature_extractor.pth'))
    
    # Save drift detector configuration
    detector_config = {
        'kernel_sigma': drift_detector.kernel.sigma if hasattr(drift_detector.kernel, 'sigma') else None,
        'base_outputs': drift_detector.base_outputs if hasattr(drift_detector, 'base_outputs') else None,
    }
    
    torch.save(detector_config, os.path.join(save_dir, 'drift_detector.pth'))
    print(f"Models saved in {save_dir}/")
