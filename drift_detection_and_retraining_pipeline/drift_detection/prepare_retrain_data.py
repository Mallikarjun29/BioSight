#!/usr/bin/env python3

"""
Prepare retraining data by combining drifted images and original dataset.
"""

import logging
import argparse
from pathlib import Path
import torch
from torch.utils.data import DataLoader, Dataset, ConcatDataset
from torchvision import transforms
from PIL import Image
from tqdm import tqdm

from database import DriftDatabase
from prepare_data import DataPreparation

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# Path mapping for converting Docker paths to local paths
DOCKER_PATH_PREFIX = Path("/app/biosight")
LOCAL_PATH_PREFIX = Path(__file__).parent.parent / "client-side/biosight"


class DriftedImagesDataset(Dataset):
    """Dataset for drifted images fetched from the database."""
    
    def __init__(self, image_entries, docker_to_local_fn, transform=None):
        """Initialize dataset with drifted images from database.
        
        Args:
            image_entries: List of image entries from database
            docker_to_local_fn: Function to convert Docker paths to local paths
            transform: Transforms to apply to images
        """
        self.transform = transform
        self.entries = []
        
        # Filter valid entries (same approach as in check_drift_batch.py)
        for entry in image_entries:
            # Convert Docker path to local path
            local_path = docker_to_local_fn(entry['upload_path'])
            
            if local_path.exists():
                self.entries.append({
                    'path': local_path,
                    'filename': entry['original_filename'],
                    'class': entry['predicted_class']
                })
    
    def __len__(self):
        return len(self.entries)
    
    def __getitem__(self, idx):
        entry = self.entries[idx]
        try:
            image = Image.open(entry['path']).convert('RGB')
            if self.transform:
                image = self.transform(image)
            return image, entry['class']
        except Exception as e:
            logger.error(f"Error loading image {entry['path']} for {entry['filename']}: {e}")
            # Return a black image as fallback
            if self.transform:
                return torch.zeros((3, 224, 224)), entry['class']
            else:
                return Image.new('RGB', (224, 224), color='black'), entry['class']


def convert_docker_path_to_local(docker_path):
    """Convert Docker container path to local filesystem path.
    
    Args:
        docker_path: Path in Docker container format
    
    Returns:
        Path object pointing to local filesystem equivalent
    """
    docker_path = Path(docker_path)
    try:
        relative_path = docker_path.relative_to(DOCKER_PATH_PREFIX)
        return LOCAL_PATH_PREFIX / relative_path
    except ValueError:
        return docker_path  # Return original if not relative to prefix


def prepare_retraining_data(original_data_dir, max_drifted_images=100, batch_size=32, num_workers=4):
    """
    Prepare combined dataset for retraining with drifted and original images.
    
    Args:
        original_data_dir: Path to original dataset directory
        max_drifted_images: Maximum number of drifted images to include
        batch_size: Batch size for data loaders
        num_workers: Number of worker processes for data loading
    
    Returns:
        Tuple of DataLoaders (train, val, test)
    """
    # Define image transforms
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # 1. Get drifted images from the database
    db = DriftDatabase()
    drifted_image_entries = []
    
    try:
        if not db.connect():
            logger.error("Failed to connect to the database.")
            return None, None, None
        
        # Fetch drifted images, excluding 'unknown' class
        # Include upload_path field (crucial for path conversion)
        drifted_image_entries = list(db.collection.find(
            {
                'drift_detected': True,
                'used_in_training': False,
                'predicted_class': {'$ne': 'unknown'}  # Exclude unknown class
            },
            {
                'original_filename': 1,
                'predicted_class': 1, 
                'upload_path': 1,  # Important for locating files
                '_id': 0
            }
        ).sort('drift_checked_at', -1).limit(max_drifted_images))
        
        logger.info(f"Fetched {len(drifted_image_entries)} drifted images from database")
    except Exception as e:
        logger.error(f"Error fetching drifted images: {e}")
    finally:
        db.close()
    
    # 2. Prepare original dataset
    data_prep = DataPreparation(
        original_data_dir,
        batch_size=batch_size,
        num_workers=num_workers
    )
    
    # Get original data loaders
    train_loader, val_loader, test_loader = data_prep.get_data_loaders()
    
    # If no drifted images, return original loaders
    if not drifted_image_entries:
        logger.warning("No drifted images found. Using only original dataset.")
        return train_loader, val_loader, test_loader
    
    # 3. Create dataset for drifted images using the same approach as check_drift_batch.py
    drifted_dataset = DriftedImagesDataset(
        drifted_image_entries,
        convert_docker_path_to_local,
        transform
    )
    
    if len(drifted_dataset) == 0:
        logger.warning("Could not locate any drifted image files. Using only original dataset.")
        return train_loader, val_loader, test_loader
    
    logger.info(f"Successfully located {len(drifted_dataset)} drifted image files")
    
    # 4. Create class mapping from name to index
    class_to_idx = {cls_name: idx for idx, cls_name in enumerate(data_prep.classes)}
    
    # Apply class mapping to drifted dataset entries
    mapped_entries = []
    for i in range(len(drifted_dataset)):
        img, class_name = drifted_dataset[i]
        if class_name in class_to_idx:
            mapped_entries.append((img, class_to_idx[class_name]))
        else:
            logger.warning(f"Class '{class_name}' not found in original dataset. Skipping.")
    
    # Create a simple dataset from mapped entries
    class MappedDataset(Dataset):
        def __init__(self, entries):
            self.entries = entries
        
        def __len__(self):
            return len(self.entries)
        
        def __getitem__(self, idx):
            return self.entries[idx]
    
    mapped_drifted_dataset = MappedDataset(mapped_entries)
    
    # 5. Combine datasets for training
    original_train_dataset = train_loader.dataset
    combined_dataset = ConcatDataset([original_train_dataset, mapped_drifted_dataset])
    
    # 6. Create combined training loader
    combined_train_loader = DataLoader(
        combined_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    
    # 7. Update training status in database
    update_training_status(drifted_image_entries)
    
    logger.info(f"Created combined dataset with {len(combined_dataset)} images "
                f"({len(original_train_dataset)} original + {len(mapped_drifted_dataset)} drifted)")
    
    return combined_train_loader, val_loader, test_loader


def update_training_status(drifted_images):
    """Mark drifted images as used in training."""
    db = DriftDatabase()
    
    try:
        if not db.connect():
            logger.error("Failed to connect to the database.")
            return False
        
        for image in drifted_images:
            filename = image.get('original_filename')
            if filename:
                db.collection.update_one(
                    {'original_filename': filename},
                    {'$set': {'used_in_training': True}}
                )
        
        logger.info(f"Updated training status for {len(drifted_images)} drifted images")
    except Exception as e:
        logger.error(f"Error updating training status: {e}")
    finally:
        db.close()


def main():
    """Main function to prepare retraining data."""
    parser = argparse.ArgumentParser(description='Prepare combined dataset for retraining')
    parser.add_argument('--data-dir', type=str, default='inaturalist_12K',
                        help='Path to original dataset directory')
    parser.add_argument('--max-drifted', type=int, default=100,
                        help='Maximum number of drifted images to include')
    parser.add_argument('--batch-size', type=int, default=32,
                        help='Batch size for data loaders')
    parser.add_argument('--num-workers', type=int, default=4,
                        help='Number of worker processes for data loading')
    
    args = parser.parse_args()
    
    # Create combined dataset and data loaders
    train_loader, val_loader, test_loader = prepare_retraining_data(
        original_data_dir=args.data_dir,
        max_drifted_images=args.max_drifted,
        batch_size=args.batch_size,
        num_workers=args.num_workers
    )
    
    if train_loader:
        # Print information about the dataset
        print(f"\nCombined training dataset size: {len(train_loader.dataset)} images")
        print(f"Validation dataset size: {len(val_loader.dataset)} images")
        print(f"Test dataset size: {len(test_loader.dataset)} images")
        
        # Sample a batch to verify
        for images, labels in train_loader:
            print(f"\nBatch shape: {images.shape}")
            print(f"Labels shape: {labels.shape}")
            print(f"Sample labels: {labels[:5]}")
            break


if __name__ == '__main__':
    main()