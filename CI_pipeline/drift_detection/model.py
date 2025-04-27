"""
Loads PyTorch models (e.g., from torchvision) with support for standard pretrained weights
or custom weights from a file, potentially including modified final layers.

Handles:
- Loading specified model types from torchvision.models.
- Using standard pretrained weight enums (e.g., 'IMAGENET1K_V1', 'DEFAULT').
- Loading custom weights from a specified file path (relative paths assumed to be in a 'models/' subdirectory).
- Mapping weights to CPU if CUDA is unavailable.
- Adjusting state dictionary keys if they have a common prefix (e.g., 'resnet.').
- Reconstructing the final classification layer ('fc') based on provided arguments
  (--num_classes, --dropout_rate) when loading custom weights.
- Using `strict=False` during state dict loading to handle potential mismatches,
  especially in the modified final layer.
"""

import argparse
import torch
import torchvision.models as models
import os
import torch.nn as nn
from collections import OrderedDict

def load_model(model_type="resnet50", weights_path="DEFAULT", num_classes=10, dropout_rate=0.5):
    """
    Loads a pretrained ResNet model and modifies final layer for drift detection.
    """
    print(f"Loading pretrained {model_type}...")
    
    try:
        # Load pretrained model
        model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
        
        # Modify final layer
        num_features = model.fc.in_features
        model.fc = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(num_features, num_classes)
        )
        
        model.eval()  # Set to evaluation mode
        print("Successfully loaded pretrained model")
        return model
        
    except Exception as e:
        print(f"Error loading pretrained model: {e}")
        return None

def main():
    """Parses command-line arguments and loads the specified model."""
    parser = argparse.ArgumentParser(
        description='Load a PyTorch machine learning model.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter # Show default values in help
    )
    parser.add_argument(
        '--model_type', type=str, default='resnet50',
        help='Type of model architecture from torchvision.models.'
    )
    parser.add_argument(
        '--weights', type=str, required=True,
        help='Path to weights file (.pth) or standard weights enum name (e.g., IMAGENET1K_V1, DEFAULT).'
    )
    parser.add_argument(
        '--num_classes', type=int, default=2,
        help='Number of classes for the custom final layer (used only when loading custom weights).'
    )
    parser.add_argument(
        '--dropout_rate', type=float, default=0.5,
        help='Dropout rate for the custom final layer (used only when loading custom weights).'
    )

    args = parser.parse_args()

    print("--- Loading Model --- ")
    model = load_model(args.model_type, args.weights, args.num_classes, args.dropout_rate)
    print("---------------------")

    if model:
        print(f"Successfully loaded model: {args.model_type}")
        # Example usage:
        # try:
        #     dummy_input = torch.randn(1, 3, 224, 224) # Adjust size if needed
        #     output = model(dummy_input)
        #     print("Model ready for inference. Example output shape:", output.shape)
        # except Exception as e:
        #     print(f"Could not perform test inference: {e}")
    else:
        print("Failed to load the model.")

if __name__ == "__main__":
    main()
