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

def load_model(model_type, weights_path, num_classes=1000, dropout_rate=0.5):
    """
    Loads or constructs a PyTorch model based on type and weights specification.

    Args:
        model_type (str): Name of the model architecture in torchvision.models (e.g., 'resnet50').
        weights_path (str): Identifier for weights. Can be:
            - A standard weights enum name (e.g., 'IMAGENET1K_V1', 'DEFAULT').
            - A file path to custom weights (.pth).
        num_classes (int): Number of output classes for the final layer if loading custom weights
                           and modifying the classifier.
        dropout_rate (float): Dropout rate for the custom final layer if modifying the classifier.

    Returns:
        torch.nn.Module or None: The loaded PyTorch model in evaluation mode, or None if loading fails.
    """
    print(f"Loading model type: {model_type}")

    try:
        model_constructor = getattr(models, model_type)
    except AttributeError:
        print(f"Error: Model type '{model_type}' not found in torchvision.models.")
        available_models = [name for name in dir(models) if callable(getattr(models, name)) and not name.startswith("_")]
        print("Available models:", available_models)
        return None

    # Determine if weights_path corresponds to a standard weights enum
    weights_enum_name = None
    weights_enum = None
    try:
        # Attempt to find the corresponding Weights enum class (e.g., ResNet50_Weights)
        enum_class_name = f"{model_type.capitalize().replace('_', '')}_Weights"
        if hasattr(models, enum_class_name):
            weights_enum_class = getattr(models, enum_class_name)
            for weight_name in dir(weights_enum_class):
                if weight_name.upper() == weights_path.upper() and not weight_name.startswith("_"):
                    weights_enum_name = weight_name
                    weights_enum = getattr(weights_enum_class, weight_name)
                    break
        elif weights_path.upper() == 'DEFAULT': # Handle 'DEFAULT' explicitly
             weights_enum = 'DEFAULT'
             weights_enum_name = 'DEFAULT'
    except Exception as e:
        # This might happen if the model doesn't follow the Weights naming convention
        print(f"Info: Could not automatically determine standard weights enum for {model_type}. Assuming file path. ({e})")

    # Load standard pretrained model or custom model from file
    if weights_enum:
        print(f"Loading standard pretrained weights: {weights_enum_name}")
        try:
            model = model_constructor(weights=weights_enum)
            print("Standard pretrained model loaded successfully.")
        except Exception as e:
             print(f"Error loading model {model_type} with standard weights {weights_enum_name}: {e}")
             return None
    else:
        # Handle loading from a custom weights file
        print(f"Attempting to load custom model structure and weights from file: {weights_path}")
        if not os.path.isabs(weights_path):
            weights_path = os.path.join("models", weights_path)
            print(f"Relative path detected. Looking for weights in: {weights_path}")

        try:
            # Instantiate the base model architecture without pretrained weights
            model = model_constructor(weights=None)

            # Modify the final layer if it's a standard classifier structure
            if hasattr(model, 'fc') and isinstance(model.fc, nn.Linear):
                print(f"Modifying final classifier layer for {num_classes} classes with dropout {dropout_rate}.")
                num_features = model.fc.in_features
                model.fc = nn.Sequential(
                    nn.Dropout(dropout_rate),
                    nn.Linear(num_features, 512),
                    nn.ReLU(),
                    nn.Dropout(dropout_rate),
                    nn.Linear(512, num_classes)
                )
            else:
                print(f"Warning: Model type '{model_type}' might not have a standard 'fc' layer. Attempting to load weights into the original structure.")

            # Load the state dictionary, mapping to CPU if needed
            map_location = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
            state_dict = torch.load(weights_path, map_location=map_location)

            # Adjust state_dict keys if they have a common prefix (e.g., 'resnet.')
            new_state_dict = OrderedDict()
            # Check if any key starts with a common model prefix like 'resnet.' or 'module.'
            prefix_to_remove = None
            common_prefixes = ['resnet.', 'module.'] # Add other potential prefixes if needed
            for prefix in common_prefixes:
                 if any(k.startswith(prefix) for k in state_dict.keys()):
                      prefix_to_remove = prefix
                      print(f"Detected state_dict keys with prefix '{prefix_to_remove}'. Removing it.")
                      break

            for k, v in state_dict.items():
                if prefix_to_remove and k.startswith(prefix_to_remove):
                    name = k[len(prefix_to_remove):] # remove prefix
                    new_state_dict[name] = v
                else:
                    new_state_dict[k] = v

            # Load the potentially modified state dict into the potentially modified model structure
            # Use strict=False to tolerate mismatches (e.g., in the modified fc layer)
            incompatible_keys = model.load_state_dict(new_state_dict, strict=False)
            if incompatible_keys.missing_keys or incompatible_keys.unexpected_keys:
                print("Warning: State dict loading encountered mismatches (strict=False used).")
                if incompatible_keys.missing_keys:
                    print(f"  Missing keys in model: {incompatible_keys.missing_keys}")
                if incompatible_keys.unexpected_keys:
                    print(f"  Unexpected keys in state_dict: {incompatible_keys.unexpected_keys}")
            print("Model loaded successfully with custom weights.")

        except FileNotFoundError:
            print(f"Error: Weights file not found at {weights_path}")
            return None
        except Exception as e:
            print(f"Error loading weights from file {weights_path}: {e}")
            return None

    model.eval()  # Set the model to evaluation mode
    return model

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
