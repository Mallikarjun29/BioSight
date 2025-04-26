"""Dataset handling utilities for iNaturalist dataset including validation and data loading."""

import os
import torch
import numpy as np
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset, random_split
from sklearn.model_selection import train_test_split

class INaturalistDataset:
    """Dataset validation and inspection utilities.
    
    Provides tools to validate dataset structure and inspect image files.
    """
    
    def __init__(self, data_dir):
        """Initialize dataset validator.
        
        Args:
            data_dir (str): Root directory containing dataset
        """
        self.data_dir = data_dir

    def inspect_directory(self, dir_path):
        """Analyze directory contents and file extensions.
        
        Args:
            dir_path (str): Path to directory for inspection
        """
        print(f"\nInspecting directory: {dir_path}")
        if not os.path.exists(dir_path):
            print("Directory does not exist!")
            return

        files = os.listdir(dir_path)
        if not files:
            print("Directory is empty!")
            return

        print(f"Found {len(files)} items in directory")
        extensions = {}
        for file in files:
            full_path = os.path.join(dir_path, file)
            if os.path.isfile(full_path):
                ext = os.path.splitext(file)[1].lower()
                extensions[ext] = extensions.get(ext, 0) + 1

        print("\nFile extensions found:")
        for ext, count in extensions.items():
            print(f"{ext}: {count} files")

        if not extensions:
            print("No files found (might be subdirectories)")
            # List directory contents
            print("\nDirectory contents:")
            for item in files:
                print(f"- {item} ({'dir' if os.path.isdir(os.path.join(dir_path, item)) else 'file'})")

    def validate_dataset_subset(self, subset_name):
        """Validate structure and contents of dataset subset.
        
        Args:
            subset_name (str): Name of subset (train/val)
            
        Returns:
            bool: True if validation passes
        """
        subset_dir = os.path.join(self.data_dir, subset_name)
        print(f"\nValidating {subset_name} dataset structure at: {subset_dir}")
        valid_extensions = ('.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tif', '.tiff', '.webp')

        if not os.path.exists(subset_dir):
            print(f"Error: {subset_name} directory not found at {subset_dir}")
            return False

        all_classes_valid = True
        class_folders = [d for d in os.listdir(subset_dir) if os.path.isdir(os.path.join(subset_dir, d))]
        if not class_folders:
             print(f"Error: No class subdirectories found in {subset_dir}")
             return False

        print(f"Found {len(class_folders)} potential class folders.")

        for class_name in class_folders:
            class_dir = os.path.join(subset_dir, class_name)
            if not os.path.isdir(class_dir): # Should not happen based on above check, but good practice
                continue

            valid_files_found = False
            image_files = []
            for file in os.listdir(class_dir):
                if file.lower().endswith(valid_extensions):
                    valid_files_found = True
                    image_files.append(file)

            if not valid_files_found:
                print(f"\nWarning: No valid images found in class '{class_name}' directory")
                print(f"Path: {class_dir}")
                print(f"Supported extensions are: {', '.join(valid_extensions)}")
                self.inspect_directory(class_dir) # Show contents if no valid images
                all_classes_valid = False

        if all_classes_valid:
             print(f"{subset_name} dataset structure appears valid.")
        else:
             print(f"{subset_name} dataset structure has issues (see warnings above).")
        return all_classes_valid


class DataPreparation:
    """Data preparation utilities for model training.
    
    Handles data loading, preprocessing, and train/val/test splitting.
    """
    
    def __init__(self, data_dir, batch_size=32, num_workers=4, val_split=0.2):
        """Initialize data preparation.
        
        Args:
            data_dir (str): Dataset root directory
            batch_size (int): Batch size for dataloaders 
            num_workers (int): Number of worker processes
            val_split (float): Validation set proportion
        """
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.val_split = val_split # Proportion of original train data to use for validation

        # Define data transformations (consistent for train, val, test)
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def get_data_loaders(self):
        """Create data loaders for model training.
        
        Returns:
            tuple: (train_loader, val_loader, test_loader)
            
        Raises:
            FileNotFoundError: If dataset directories not found
        """
        original_train_dir = os.path.join(self.data_dir, 'train')
        test_dir = os.path.join(self.data_dir, 'val') # Original 'val' is now 'test'

        if not os.path.exists(original_train_dir):
            raise FileNotFoundError(f"Original training directory not found: {original_train_dir}")
        if not os.path.exists(test_dir):
            raise FileNotFoundError(f"Test directory (original val) not found: {test_dir}")

        # Load the original training dataset
        original_train_dataset = datasets.ImageFolder(
            original_train_dir,
            transform=self.transform
        )

        # Perform stratified split on the original training data
        targets = original_train_dataset.targets
        train_indices, val_indices = train_test_split(
            np.arange(len(targets)),
            test_size=self.val_split,
            shuffle=True,
            stratify=targets
        )

        # Create new train and validation datasets using Subset
        new_train_dataset = Subset(original_train_dataset, train_indices)
        new_val_dataset = Subset(original_train_dataset, val_indices)

        # Load the test dataset (original validation set)
        test_dataset = datasets.ImageFolder(
            test_dir,
            transform=self.transform
        )

        print(f"Original training data size: {len(original_train_dataset)}")
        print(f"New training data size: {len(new_train_dataset)}")
        print(f"New validation data size: {len(new_val_dataset)}")
        print(f"Test data size: {len(test_dataset)}")

        # Create data loaders
        train_loader = DataLoader(
            new_train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True # Helps speed up data transfer to GPU
        )

        val_loader = DataLoader(
            new_val_dataset,
            batch_size=self.batch_size,
            shuffle=False, # No need to shuffle validation data
            num_workers=self.num_workers,
            pin_memory=True
        )

        test_loader = DataLoader(
            test_dataset,
            batch_size=self.batch_size,
            shuffle=False, # No need to shuffle test data
            num_workers=self.num_workers,
            pin_memory=True
        )

        # Store class names (should be consistent across splits)
        self.classes = original_train_dataset.classes

        return train_loader, val_loader, test_loader


if __name__ == "__main__":
    """Validate dataset structure and test data loading."""
    data_directory = "../inaturalist_12K"
    # Use the helper class for validation checks
    dataset_validator = INaturalistDataset(data_directory)

    print("\n--- Dataset Structure Validation ---")
    dataset_validator.validate_dataset_subset('train') # Validate original train folder structure
    dataset_validator.validate_dataset_subset('val')   # Validate original val (now test) folder structure
    print("--- Validation Complete ---")

    try:
        # Get the loaders (which performs the split)
        data_preparation = DataPreparation(data_directory, batch_size=32, val_split=0.2)
        train_loader, val_loader, test_loader = data_preparation.get_data_loaders()

        print(f"\nNumber of NEW training batches: {len(train_loader)}")
        print(f"Number of NEW validation batches: {len(val_loader)}")
        print(f"Number of TEST batches: {len(test_loader)}")

        print("\nClass labels (should be consistent):")
        # Access classes from the DataPreparation object after loading
        if hasattr(data_preparation, 'classes'):
             print(data_preparation.classes)
        else:
             print("Could not access class list.")

        # Optional: Iterate through a batch to check shapes
        print("\nChecking a batch from train_loader:")
        for images, labels in train_loader:
            print(f"Image batch shape: {images.shape}") # Should be [batch_size, 3, 224, 224]
            print(f"Label batch shape: {labels.shape}") # Should be [batch_size]
            break # Only check one batch

    except Exception as e:
        print(f"\nError during data loading or splitting: {str(e)}")
        import traceback
        traceback.print_exc()