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
    parser.add_argument('--tracking_uri', type=str, default=None,
                        help='MLflow tracking URI')
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
    """Train the model using specified configuration.
    
    Args:
        model (nn.Module): ResNet model to train
        train_loader (DataLoader): Training data loader
        val_loader (DataLoader): Validation data loader
        args (argparse.Namespace): Training arguments
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    criterion = nn.CrossEntropyLoss()
    params_to_optimize = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = optim.Adam(params_to_optimize, lr=args.learning_rate)

    model = model.to(device)
    criterion = criterion.to(device)
    mlflow.pytorch.autolog()
    best_val_acc = 0.0

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

        mlflow.log_metrics({
            "train_loss": running_loss/len(train_loader),
            "train_acc": train_acc,
            "val_loss": val_loss/len(val_loader),
            "val_acc": val_acc
        }, step=epoch)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), 'best_model_resnet.pth')
            mlflow.pytorch.log_model(model, "best_model_resnet")

if __name__ == "__main__":
    args = parse_args()
    mlflow.set_tracking_uri(args.tracking_uri)
    mlflow.start_run()
    mlflow.set_experiment(args.experiment_name)
    
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