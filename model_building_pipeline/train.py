"""Training script for ResNet50 model with transfer learning strategies."""

import torch
import torch.nn as nn
import torch.optim as optim
import mlflow
import mlflow.pytorch
import argparse
from prepare_data import DataPreparation
from model import ResNetModel
import numpy as np
import os

def parse_args():
    """Parse command line arguments for model training.
    
    Returns:
        argparse.Namespace: Parsed arguments
    """
    parser = argparse.ArgumentParser(description='Train ResNet model with transfer learning')
    
    parser.add_argument('--experiment_name', type=str, default='resnet_training',
                        help='MLflow experiment name')
    parser.add_argument('-e', '--epochs', type=int, default=10,
                        help='Number of training epochs')
    parser.add_argument('-b', '--batch_size', type=int, default=32,
                        help='Training batch size')
    parser.add_argument('-lr', '--learning_rate', type=float, default=0.001,
                        help='Learning rate')
    parser.add_argument('--freeze_strategy', type=str, 
                        choices=['none', 'upto_stage_1', 'upto_stage_2', 'upto_stage_3'],
                        default='none',
                        help='Layer freezing strategy')
    parser.add_argument('--dropout_rate', type=float, default=0.5,
                        help='Dropout rate')
    
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
    mlflow.set_experiment(args.experiment_name)  # Set experiment BEFORE start_run
    with mlflow.start_run(): # Start the run
        # Log parameters (optional, but good practice)
        mlflow.log_params(vars(args))
        
        data_directory = "../inaturalist_12K"
        data_preparation = DataPreparation(data_directory, batch_size=args.batch_size)
        train_loader, val_loader, test_loader = data_preparation.get_data_loaders()
        
        freeze_stage = None
        if args.freeze_strategy == 'upto_stage_1':
            freeze_stage = 1
        elif args.freeze_strategy == 'upto_stage_2':
            freeze_stage = 2
        elif args.freeze_strategy == 'upto_stage_3':
            freeze_stage = 3
        
        model = ResNetModel(
            num_classes=10,
            dropout_rate=args.dropout_rate,
            freeze_upto_stage=freeze_stage
        )
        
        train_model(model, train_loader, val_loader, args)