"""
model.py

This module defines the ResNetModel class, which is a wrapper around the ResNet50 architecture from torchvision.
The model allows for fine-tuning with options to freeze specific layers or stages of the network.

Classes:
    ResNetModel: A customizable ResNet50-based model with options for dropout and layer freezing.

Example Usage:
    # Initialize the model with specific configurations
    model = ResNetModel(num_classes=10, dropout_rate=0.5, freeze_upto_stage=2)
    output = model(torch.randn(1, 3, 224, 224))
"""

import torch
import torch.nn as nn
from torchvision.models import resnet50, ResNet50_Weights

class ResNetModel(nn.Module):
    def __init__(self,
                 num_classes=10,
                 dropout_rate=0.5,
                 freeze_upto_stage=None):
        """
        Initializes the ResNetModel.

        Args:
            num_classes (int): Number of output classes for the final classification layer.
            dropout_rate (float): Dropout rate to use in the fully connected layers.
            freeze_upto_stage (int or None): Specifies up to which stage the layers should be frozen.
                None means all layers are trainable.
        """
        super(ResNetModel, self).__init__()

        # Load pretrained ResNet50
        self.resnet = resnet50(weights=ResNet50_Weights.DEFAULT)

        # --- Freezing Logic ---
        if freeze_upto_stage is not None:
            for name, child in self.resnet.named_children():
                if name in ['conv1', 'bn1', 'relu', 'maxpool']:
                    for param in child.parameters():
                        param.requires_grad = False
                elif name.startswith('layer') and int(name.replace('layer', '')) <= freeze_upto_stage:
                    for param in child.parameters():
                        param.requires_grad = False
                else:
                    for param in child.parameters():
                        param.requires_grad = True
        else:
            for param in self.resnet.parameters():
                param.requires_grad = True
        # --- End Freezing Logic ---

        # Get number of features from last layer
        num_features = self.resnet.fc.in_features

        # Replace final fully connected layer
        self.resnet.fc = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(num_features, 512),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(512, num_classes)
        )

        # Ensure the new classifier head is trainable
        for param in self.resnet.fc.parameters():
            param.requires_grad = True

    def forward(self, x):
        """
        Forward pass of the model.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, 3, height, width).

        Returns:
            torch.Tensor: Output tensor after passing through the model.
        """
        return self.resnet(x)

if __name__ == "__main__":
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Example: Freeze up to Layer 2
    model_freeze_2 = ResNetModel(
        num_classes=10,
        dropout_rate=0.5,
        freeze_upto_stage=2
    ).to(device)

    trainable_params_freeze_2 = sum(p.numel() for p in model_freeze_2.parameters() if p.requires_grad)
    print(f"Trainable parameters (freeze up to stage 2): {trainable_params_freeze_2:,}")

    # Example: Fine-tune all layers
    model_fine_tune_all = ResNetModel(
        num_classes=10,
        dropout_rate=0.5,
        freeze_upto_stage=None
    ).to(device)

    trainable_params_fine_tune_all = sum(p.numel() for p in model_fine_tune_all.parameters() if p.requires_grad)
    print(f"Trainable parameters (fine-tune all): {trainable_params_fine_tune_all:,}")