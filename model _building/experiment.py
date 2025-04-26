"""
sweep_experiment.py

This script performs hyperparameter optimization for the ResNetModel using MLflow.
It includes functions for training the model, logging metrics, and running hyperparameter optimization
with different configurations using hyperopt.

Functions:
    train_model: Trains the ResNetModel with specified configurations and logs metrics.
    sweep_train: Sets up the hyperparameter optimization and runs the training process.

Example Usage:
    # Run the script to start hyperparameter optimization
    python sweep_experiment.py
"""

import torch
import torch.nn as nn
import torch.optim as optim
import mlflow
import mlflow.pytorch
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials
import numpy as np
from prepare_data import DataPreparation
from model import ResNetModel

# Add device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

def train_model(model, train_loader, val_loader, config):
    """
    Trains the ResNetModel and evaluates it on the validation set.

    Args:
        model (nn.Module): The ResNetModel to train.
        train_loader (DataLoader): DataLoader for the training dataset.
        val_loader (DataLoader): DataLoader for the validation dataset.
        config (dict): Configuration object containing hyperparameters.

    Logs:
        Metrics such as training loss, validation loss, and accuracy to MLflow.
    """
    criterion = nn.CrossEntropyLoss()

    # --- Optimizer Setup: Only optimize trainable parameters ---
    # Filter parameters that require gradients
    params_to_optimize = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = optim.Adam(params_to_optimize, lr=config['learning_rate'])
    # --- End Optimizer Setup ---

    # Move model to GPU
    model = model.to(device)
    criterion = criterion.to(device)

    # Enable automatic logging of parameters, metrics, and models
    mlflow.pytorch.autolog()

    best_val_acc = 0.0
    num_epochs = int(config['epochs'])

    for epoch in range(num_epochs):
        # Training phase
        model.train()  # Make sure model is in training mode
        running_loss = 0.0
        correct = 0
        total = 0

        for inputs, labels in train_loader:
            # Move data to GPU
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

        # Validation phase
        model.eval()  # Switch to evaluation mode
        val_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for inputs, labels in val_loader:
                # Move data to GPU
                inputs, labels = inputs.to(device), labels.to(device)

                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()

        val_acc = 100. * correct / total

        # Log metrics to MLflow
        mlflow.log_metrics({
            "train_loss": running_loss / len(train_loader),
            "train_acc": train_acc,
            "val_loss": val_loss / len(val_loader),
            "val_acc": val_acc
        }, step=epoch)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            # Save model checkpoint only if it's the best so far
            torch.save(model.state_dict(), 'best_model_resnet.pth')
            mlflow.pytorch.log_model(model, "best_model_resnet")
    
    return best_val_acc

def sweep_train():
    """
    Sets up hyperparameter optimization and runs the training process.

    Loads the dataset, initializes the ResNetModel with different configurations,
    and trains the model using hyperopt with MLflow tracking.

    Returns:
        function: An objective function to be used by hyperopt for optimization.
    """
    # Load dataset
    data_directory = "../inaturalist_12K"
    data_preparation = DataPreparation(data_directory, batch_size=32)
    train_loader, val_loader, test_loader = data_preparation.get_data_loaders()

    def objective(params):
        with mlflow.start_run(nested=True):
            # Log parameters
            mlflow.log_params(params)
            
            # Set run name for MLflow
            mlflow.set_tag("mlflow.runName", 
                          f"freeze_{params['freeze_strategy']}_dr{params['dropout_rate']}_lr{params['learning_rate']:.5f}_ep{params['epochs']}")

            # Determine freeze_upto_stage based on config
            freeze_stage = None
            if params['freeze_strategy'] == 'upto_stage_1':
                freeze_stage = 1
            elif params['freeze_strategy'] == 'upto_stage_2':
                freeze_stage = 2
            elif params['freeze_strategy'] == 'upto_stage_3':
                freeze_stage = 3

            # Initialize model with the specified freeze strategy
            model = ResNetModel(
                num_classes=10,
                dropout_rate=params['dropout_rate'],
                freeze_upto_stage=freeze_stage
            )

            # Train the model and get validation accuracy
            val_acc = train_model(model, train_loader, val_loader, params)
            
            # Return the negative validation accuracy (hyperopt minimizes)
            return {'loss': -val_acc, 'status': STATUS_OK}

    return objective

if __name__ == "__main__":
    """
    Main entry point for the script.

    Sets up the hyperparameter search space and starts the hyperopt optimization
    with MLflow tracking.
    """
    # Set up the MLflow experiment
    mlflow.set_experiment("resnet_hyperparameter_optimization")
    
    # Define the search space
    search_space = {
        'dropout_rate': hp.choice('dropout_rate', [0.3, 0.5]),
        'freeze_strategy': hp.choice('freeze_strategy', ['none', 'upto_stage_1', 'upto_stage_2', 'upto_stage_3']),
        'learning_rate': hp.loguniform('learning_rate', np.log(1e-5), np.log(1e-3)),
        'epochs': hp.quniform('epochs', 2, 10, 1)  # Integer values between 2 and 10
    }
    
    # Create trials object to store results
    trials = Trials()
    
    # Start a parent MLflow run
    with mlflow.start_run(run_name="hyperparameter_optimization"):
        # Run hyperopt optimization
        best = fmin(
            fn=sweep_train(),
            space=search_space,
            algo=tpe.suggest,
            max_evals=10,  # Same number of trials as in the wandb version
            trials=trials
        )
        
        # Log the best parameters
        mlflow.log_params({f"best_{k}": v for k, v in best.items()})
        
        print("Best parameters:", best)