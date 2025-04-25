import os
import torch
import numpy as np # Import numpy
from PIL import Image
import torchvision.transforms as transforms
from load_drift_detector import load_drift_models # This now loads alibi-detect models
from database import DriftDatabase
from pathlib import Path
import logging
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class DriftDataset(Dataset):
    """Custom dataset for drift detection"""
    def __init__(self, image_entries, docker_to_local_fn, transform=None):
        self.entries = []
        # Filter valid entries
        for entry in image_entries:
            local_path = docker_to_local_fn(entry['upload_path'])
            if local_path.exists():
                self.entries.append({
                    'path': local_path,
                    'filename': entry['original_filename']
                })
        self.transform = transform

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, idx):
        entry = self.entries[idx]
        try:
            image = Image.open(entry['path']).convert('RGB')
            if self.transform:
                image = self.transform(image)
            return image, entry['filename']
        except Exception as e:
            logger.error(f"Error loading image {entry['path']} for {entry['filename']}: {e}")
            # Return None or a placeholder if an image fails to load
            return None, entry['filename']

# Add a collate function to handle potential None values from failed image loads
def collate_fn(batch):
    # Filter out None images
    batch = [(img, fname) for img, fname in batch if img is not None]
    if not batch:
        return None, None # Return None if the whole batch failed
    # Separate images and filenames
    images, filenames = zip(*batch)
    # Stack images into a tensor
    images = torch.stack(images, 0)
    return images, filenames

class DriftChecker:
    def __init__(self):
        # Disable gradients globally for inference
        torch.set_grad_enabled(False)
        
        # Load models using the updated loader
        self.feature_extractor, self.drift_detector, self.device = load_drift_models()
        logger.info(f"Using device: {self.device}")
        
        # Ensure feature extractor is on the correct device and in eval mode
        self.feature_extractor = self.feature_extractor.to(self.device)
        self.feature_extractor.eval()
        
        self.db = DriftDatabase()

        # Set up image transformation (should match training)
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        # REMOVED: self.DRIFT_THRESHOLD = self.drift_detector.p_val
        # The threshold is handled internally by the detector's predict method.
        # logger.info(f"Using drift threshold (p-value): {self.DRIFT_THRESHOLD}") # Also removed log
        
        # Path mapping
        self.docker_path_prefix = Path("/app/biosight")
        self.local_path_prefix = Path(__file__).parent.parent / "client-side/biosight"
        
        # Batch size for inference (can be different from training)
        self.batch_size = 4
        logger.info(f"Using inference batch size: {self.batch_size}")

    def convert_docker_path_to_local(self, docker_path: str) -> Path:
        """Convert Docker container path to local filesystem path."""
        docker_path = Path(docker_path)
        try:
            relative_path = docker_path.relative_to(self.docker_path_prefix)
            return self.local_path_prefix / relative_path
        except ValueError:
            return docker_path # Return original if not relative to prefix

    # process_image method is no longer needed as we use batch processing

    def check_recent_images(self, limit=1000):
        """Check drift on recent images from database using DataLoader and alibi-detect"""
        if not self.db.connect():
            logger.error("Failed to connect to database")
            return
        
        try:
            # Get entries from DB
            image_entries = list(self.db.collection.find(
                {'drift_checked_at': {'$exists': False}},
                {'original_filename': 1, 'upload_path': 1}
            ).sort('timestamp', -1).limit(limit))

            if not image_entries:
                logger.info("No new images found in database to check for drift.")
                return

            # Create dataset
            dataset = DriftDataset(
                image_entries,
                self.convert_docker_path_to_local,
                self.transform
            )
            
            if len(dataset) == 0:
                logger.info("No valid image files found corresponding to database entries.")
                return

            # Create DataLoader
            dataloader = DataLoader(
                dataset,
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=2,
                pin_memory=True if self.device.type == 'cuda' else False,
                collate_fn=collate_fn # Use the custom collate function
            )

            processed_count = 0
            drift_detected_count = 0
            
            # Process batches
            with torch.inference_mode(): # Use inference_mode for efficiency
                for batch_imgs, filenames in tqdm(dataloader, desc="Processing batches"):
                    # Skip if the batch is empty after collation
                    if batch_imgs is None or filenames is None:
                        logger.warning("Skipping an empty or failed batch.")
                        continue

                    # Move batch to device
                    batch_imgs = batch_imgs.to(self.device)
                    
                    # Get features (as numpy array for alibi-detect)
                    features = self.feature_extractor(batch_imgs).cpu().numpy()
                    
                    # Predict drift using alibi-detect
                    # Ensure features have the correct shape (n_samples, n_features)
                    if features.ndim == 1:
                         features = features.reshape(1, -1)
                    elif features.shape[0] == 0:
                         logger.warning("Skipping batch with 0 features.")
                         continue

                    predictions = self.drift_detector.predict(features)
                    
                    # Extract results
                    # Note: alibi-detect's predict returns drift status based on p_val < threshold
                    drift_detected = bool(predictions['data']['is_drift'])
                    p_val = float(predictions['data']['p_val'])
                    
                    # Log batch result
                    logger.info(f"Batch (size {len(filenames)}): p-value={p_val:.6f}, Drift Detected={drift_detected}")
                    
                    # Update database for all images in this batch
                    for filename in filenames:
                        self.db.update_drift_status(
                            filename=filename,
                            drift_detected=drift_detected # Apply batch result to all images in batch
                        )
                        
                        processed_count += 1
                        if drift_detected:
                            drift_detected_count += 1

            logger.info(f"Completed processing {processed_count} images")
            logger.info(f"Total images marked with drift: {drift_detected_count}") # Note: This counts images, not batches

        except Exception as e:
            logger.error(f"Error during batch processing: {e}", exc_info=True)
        finally:
            self.db.close()

if __name__ == "__main__":
    checker = DriftChecker()
    checker.check_recent_images(limit=1000)