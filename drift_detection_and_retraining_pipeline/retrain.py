"""Training script for ResNet50 model with transfer learning strategies."""

import torch
import torch.nn as nn
import torch.optim as optim
import mlflow
import mlflow.pytorch
import argparse
from torch.utils.data import DataLoader
# Import MappedDataset from prepare_retrain_data
from prepare_retrain_data import prepare_retraining_data, MappedDataset
import numpy as np
import os
import sys
from pathlib import Path
import json
import joblib
import datetime  # Import the datetime module
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Add model_building_pipeline to path if not already there
model_building_path = Path(__file__).parent.parent / "model_building_pipeline"
if str(model_building_path) not in sys.path:
    sys.path.append(str(model_building_path))

# Import ResNetModel from model_building_pipeline 
from model_building_pipeline.model import ResNetModel

def parse_args():
    """Parse command line arguments for model training.
    
    Returns:
        argparse.Namespace: Parsed arguments
    """
    parser = argparse.ArgumentParser(description='Train ResNet model with transfer learning')
    
    # Update argument names to accept both formats (with hyphen and underscore)
    parser.add_argument('--experiment-name', '--experiment_name', dest='experiment_name',
                        type=str, default='resnet_training',
                        help='MLflow experiment name')
    parser.add_argument('--tracking-uri', '--tracking_uri', dest='tracking_uri',
                        type=str, default=None,
                        help='MLflow tracking URI')
    parser.add_argument('-e', '--epochs', type=int, default=10,
                        help='Number of training epochs')
    parser.add_argument('-b', '--batch-size', '--batch_size', dest='batch_size',
                        type=int, default=32,
                        help='Training batch size')
    parser.add_argument('-lr', '--learning-rate', '--learning_rate', dest='learning_rate',
                        type=float, default=0.001,
                        help='Learning rate')
    parser.add_argument('--freeze-strategy', '--freeze_strategy', dest='freeze_strategy',
                        type=str, choices=['none', 'upto_stage_1', 'upto_stage_2', 'upto_stage_3'],
                        default='upto_stage_3',
                        help='Layer freezing strategy')
    parser.add_argument('--dropout-rate', '--dropout_rate', dest='dropout_rate',
                        type=float, default=0.5,
                        help='Dropout rate')
    parser.add_argument('--data-dir', '--data_dir', dest='data_dir',
                        type=str, default='../inaturalist_12K',
                        help='Path to original dataset directory')
    parser.add_argument('--max-drifted', '--max_drifted', dest='max_drifted',
                        type=int, default=100,
                        help='Maximum number of drifted images to include')
    parser.add_argument('--num-workers', '--num_workers', dest='num_workers',
                        type=int, default=4,
                        help='Number of worker processes for data loading')
    parser.add_argument('--load-datasets', '--load_datasets', dest='load_datasets',
                        type=str, help='Load prepared datasets from pickle file')
    
    return parser.parse_args()

def train_model(model, train_loader, val_loader, args):
    """Train the model using specified configuration."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    criterion = nn.CrossEntropyLoss()
    params_to_optimize = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = optim.Adam(params_to_optimize, lr=args.learning_rate)

    model = model.to(device)
    criterion = criterion.to(device)
    
    # Don't use autolog, we'll log manually with signature
    # mlflow.pytorch.autolog()
    
    best_val_acc = 0.0
    
    # Get a batch of data for input example
    example_inputs = None
    for inputs, _ in train_loader:
        example_inputs = inputs[:1].to('cpu')  # Just need one example
        break
    
    for epoch in range(args.epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

        train_acc = 100. * correct / total

        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()

        val_acc = 100. * correct / total

        # Log metrics manually
        mlflow.log_metrics({
            "train_loss": running_loss/len(train_loader),
            "train_acc": train_acc,
            "val_loss": val_loss/len(val_loader),
            "val_acc": val_acc
        }, step=epoch)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), 'best_model_resnet.pth')
            
            # Create model signature
            from mlflow.models.signature import infer_signature
            
            # Move model to CPU for inference
            model_cpu = model.to('cpu')
            
            # Generate output on example data
            with torch.no_grad():
                example_outputs = model_cpu(example_inputs)
            
            # Infer signature from input and output examples
            signature = infer_signature(
                example_inputs.numpy(), 
                example_outputs.numpy()
            )
            
            # Log model with signature and input example
            mlflow.pytorch.log_model(
                model_cpu,
                "best_model_resnet",
                signature=signature,
                input_example=example_inputs.numpy()
            )
            
            # Move model back to original device
            model = model.to(device)

if __name__ == "__main__":
    args = parse_args()
    
    # Set MLflow tracking URI if provided
    if args.tracking_uri:
        mlflow.set_tracking_uri(args.tracking_uri)
    
    # Set experiment BEFORE start_run (same as train.py)
    mlflow.set_experiment(args.experiment_name)
    
    # Using the context manager pattern for MLflow run
    with mlflow.start_run():
        # Log parameters (same as train.py)
        mlflow.log_params(vars(args))
        
        # Load datasets if path is provided
        if args.load_datasets:
            import joblib  # Use joblib
            datasets = joblib.load(args.load_datasets)
            
            # Create data loaders from saved datasets
            train_dataset = datasets['train']
            val_dataset = datasets['val']
            test_dataset = datasets['test']
            original_train_size = datasets['original_train_size']
            
            train_loader = DataLoader(
                train_dataset,
                batch_size=args.batch_size,
                shuffle=True,
                num_workers=args.num_workers,
                pin_memory=True
            )
            
            val_loader = DataLoader(
                val_dataset,
                batch_size=args.batch_size,
                shuffle=False,
                num_workers=args.num_workers,
                pin_memory=True
            )
            
            test_loader = DataLoader(
                test_dataset,
                batch_size=args.batch_size,
                shuffle=False,
                num_workers=args.num_workers,
                pin_memory=True
            )
        else:
            # Original code for preparing data
            train_loader, val_loader, test_loader, original_train_size = prepare_retraining_data(
                original_data_dir=args.data_dir,
                max_drifted_images=args.max_drifted,
                batch_size=args.batch_size,
                num_workers=args.num_workers
            )
        
        # Log dataset composition information to MLflow
        mlflow.log_params({
            'original_train_size': original_train_size,
            'total_train_size': len(train_loader.dataset),
            'drifted_images_count': len(train_loader.dataset) - original_train_size,
            'val_size': len(val_loader.dataset),
            'test_size': len(test_loader.dataset)
        })
        
        # Setup model freezing
        freeze_stage = None
        if args.freeze_strategy == 'upto_stage_1':
            freeze_stage = 1
        elif args.freeze_strategy == 'upto_stage_2':
            freeze_stage = 2
        elif args.freeze_strategy == 'upto_stage_3':
            freeze_stage = 3
        
        # Create the model
        model = ResNetModel(
            num_classes=10,
            dropout_rate=args.dropout_rate,
            freeze_upto_stage=freeze_stage
        )
        
        # Train the model
        train_model(model, train_loader, val_loader, args)
        
        # Create models directory if it doesn't exist
        models_dir = "models"
        os.makedirs(models_dir, exist_ok=True)
        
        # Generate a versioned model name
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        model_filename = f"best_model_resnet_{timestamp}.pth"
        model_path = os.path.join(models_dir, model_filename)
        
        # Save the model
        torch.save(model.state_dict(), model_path)
        logger.info(f"Model saved to {model_path}")
        
        # Log the model as an artifact to MLflow
        mlflow.log_artifact(model_path)