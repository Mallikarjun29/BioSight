#!/usr/bin/env python3

"""
Test script to evaluate the retrained model on test data.
This script loads a trained model and evaluates its performance on the test dataset,
logging results to MLflow.
"""

import torch
import torch.nn as nn
import argparse
import sys
import os
from pathlib import Path
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_recall_fscore_support
import matplotlib.pyplot as plt
import seaborn as sns
import mlflow
import mlflow.pytorch
import io
import json
from torch.utils.data import DataLoader

# Add drift_pipeline to path if not already there
drift_pipeline_path = Path(__file__).parent.parent / "drift_pipeline"
if str(drift_pipeline_path) not in sys.path:
    sys.path.append(str(drift_pipeline_path))

from prepare_retrain_data import prepare_retraining_data

# Add model_building_pipeline to path if not already there
model_building_path = Path(__file__).parent.parent / "model_building_pipeline"
if str(model_building_path) not in sys.path:
    sys.path.append(str(model_building_path))

# Import ResNetModel from model_building_pipeline 
from model_building_pipeline.model import ResNetModel


def parse_args():
    """Parse command line arguments for model testing.
    
    Returns:
        argparse.Namespace: Parsed arguments
    """
    parser = argparse.ArgumentParser(description='Test the trained ResNet model')
    
    parser.add_argument('--model-path', type=str, default='best_model_resnet.pth',
                        help='Path to the saved model weights')
    parser.add_argument('--run-id', type=str, default=None,
                        help='MLflow run ID to associate results with')
    parser.add_argument('--experiment-name', type=str, default='resnet_testing',
                        help='MLflow experiment name')
    parser.add_argument('--tracking-uri', type=str, default=None,
                        help='MLflow tracking URI')
    parser.add_argument('-b', '--batch-size', type=int, default=32,
                        help='Testing batch size')
    parser.add_argument('--data-dir', type=str, default='../inaturalist_12K',
                        help='Path to original dataset directory')
    parser.add_argument('--max-drifted', type=int, default=100,
                        help='Maximum number of drifted images to include')
    parser.add_argument('--num-workers', type=int, default=4,
                        help='Number of worker processes for data loading')
    parser.add_argument('--num-classes', type=int, default=10,
                        help='Number of classes in the dataset')
    parser.add_argument('--dropout-rate', type=float, default=0.5,
                        help='Dropout rate used in the model')
    parser.add_argument('--load-datasets', type=str,
                      help='Load prepared datasets from pickle file')
    
    return parser.parse_args()


def evaluate_model(model, test_loader, device, class_names=None):
    """Evaluate the model on test data.
    
    Args:
        model: The trained model to evaluate
        test_loader: DataLoader for test data
        device: Device to run evaluation on
        class_names: Names of classes for the confusion matrix
    
    Returns:
        dict: Dictionary containing evaluation metrics
    """
    model.eval()
    criterion = nn.CrossEntropyLoss()
    
    test_loss = 0.0
    correct = 0
    total = 0
    
    # For storing predictions and actual labels
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            
            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            
            # Store predictions and targets for confusion matrix
            all_preds.extend(predicted.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())
    
    # Calculate metrics
    accuracy = 100.0 * correct / total
    avg_loss = test_loss / len(test_loader)
    
    # Calculate precision, recall, and F1 score
    precision, recall, f1, _ = precision_recall_fscore_support(
        all_targets, all_preds, average='weighted'
    )
    
    # Create classification report
    report = classification_report(
        all_targets, 
        all_preds, 
        target_names=class_names if class_names else None,
        output_dict=True
    )
    
    # Create confusion matrix
    conf_matrix = confusion_matrix(all_targets, all_preds)
    
    return {
        'accuracy': accuracy,
        'avg_loss': avg_loss,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'classification_report': report,
        'confusion_matrix': conf_matrix,
        'all_preds': all_preds,
        'all_targets': all_targets
    }


def plot_confusion_matrix(conf_matrix, class_names):
    """Create confusion matrix plot and return it as a figure.
    
    Args:
        conf_matrix: Confusion matrix
        class_names: Names of the classes
    
    Returns:
        matplotlib.figure.Figure: The confusion matrix figure
    """
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(
        conf_matrix, 
        annot=True, 
        fmt='d', 
        cmap='Blues',
        xticklabels=class_names,
        yticklabels=class_names,
        ax=ax
    )
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    
    return fig


def log_metrics_to_mlflow(metrics, class_names=None):
    """Log evaluation metrics to MLflow.
    
    Args:
        metrics: Dictionary containing evaluation metrics
        class_names: Names of the classes
    """
    # Log basic metrics
    mlflow.log_metrics({
        'test_accuracy': metrics['accuracy'],
        'test_loss': metrics['avg_loss'],
        'test_precision': metrics['precision'],
        'test_recall': metrics['recall'],
        'test_f1': metrics['f1']
    })
    
    # Log per-class metrics
    report = metrics['classification_report']
    for cls, vals in report.items():
        if isinstance(vals, dict):
            cls_name = class_names[int(cls)] if class_names and cls.isdigit() else cls
            mlflow.log_metrics({
                f'class_{cls_name}_precision': vals['precision'],
                f'class_{cls_name}_recall': vals['recall'],
                f'class_{cls_name}_f1': vals['f1-score'],
                f'class_{cls_name}_support': vals['support']
            })
    
    # Create and log confusion matrix image
    if class_names:
        fig = plot_confusion_matrix(metrics['confusion_matrix'], class_names)
        buf = io.BytesIO()
        fig.savefig(buf, format='png')
        buf.seek(0)
        mlflow.log_figure(fig, "confusion_matrix.png")
        plt.close(fig)
    
    # Save the confusion matrix as a CSV and log it as an artifact
    np.savetxt("confusion_matrix.csv", metrics['confusion_matrix'], delimiter=",")
    mlflow.log_artifact("confusion_matrix.csv")
    
    # Save the detailed classification report as JSON and log it
    detailed_report = {
        'accuracy': metrics['accuracy'],
        'loss': metrics['avg_loss'],
        'precision': metrics['precision'],
        'recall': metrics['recall'],
        'f1': metrics['f1'],
        'class_metrics': {}
    }
    
    # Add per-class metrics to the report
    for cls, vals in report.items():
        if isinstance(vals, dict) and cls.isdigit():
            cls_name = class_names[int(cls)] if class_names else cls
            detailed_report['class_metrics'][cls_name] = {
                'precision': vals['precision'],
                'recall': vals['recall'],
                'f1': vals['f1-score'],
                'support': vals['support']
            }
    
    # Save as JSON and log it
    with open('evaluation_report.json', 'w') as f:
        json.dump(detailed_report, f, indent=4)
    
    mlflow.log_artifact('evaluation_report.json')


def main():
    """Main function to test the model."""
    args = parse_args()
    
    # Set MLflow tracking
    if args.tracking_uri:
        mlflow.set_tracking_uri(args.tracking_uri)
    
    mlflow.set_experiment(args.experiment_name)
    
    # Start MLflow run
    with mlflow.start_run(run_id=args.run_id):
        # Log parameters
        mlflow.log_params({
            'model_path': args.model_path,
            'data_dir': args.data_dir,
            'batch_size': args.batch_size,
            'max_drifted_images': args.max_drifted,
            'num_classes': args.num_classes,
            'dropout_rate': args.dropout_rate
        })
        
        # Set device for testing
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {device}")
        mlflow.log_param('device', device.type)
        
        # Load datasets if path is provided
        if args.load_datasets:
            import pickle
            with open(args.load_datasets, 'rb') as f:
                datasets = pickle.load(f)
            
            # Create test loader from saved dataset
            test_dataset = datasets['test']
            test_loader = DataLoader(
                test_dataset,
                batch_size=args.batch_size if hasattr(args, 'batch_size') else 32,
                shuffle=False,
                num_workers=args.num_workers if hasattr(args, 'num_workers') else 4,
                pin_memory=True
            )
        else:
            # Original code for loading test data
            _, _, test_loader = prepare_retraining_data(
                original_data_dir=args.data_dir,
                max_drifted_images=args.max_drifted,
                batch_size=args.batch_size,
                num_workers=args.num_workers
            )
        
        # Initialize the model with the same architecture used during training
        model = ResNetModel(
            num_classes=args.num_classes,
            dropout_rate=args.dropout_rate
        )
        
        # Load trained weights
        if os.path.exists(args.model_path):
            model.load_state_dict(torch.load(args.model_path, map_location=device))
            print(f"Model weights loaded from {args.model_path}")
        else:
            print(f"Model weights not found at {args.model_path}")
            return
        
        model = model.to(device)
        
        # Get class names if available
        class_names = None
        try:
            from drift_pipeline.prepare_data import DataPreparation
            data_prep = DataPreparation(args.data_dir)
            # Access class names through one of the data loaders
            _, _, _ = data_prep.get_data_loaders()
            class_names = data_prep.classes
            print(f"Found {len(class_names)} classes: {class_names}")
            mlflow.log_param('class_names', class_names)
        except Exception as e:
            print(f"Could not retrieve class names: {e}")
        
        # Evaluate the model
        print("Evaluating model on test dataset...")
        metrics = evaluate_model(model, test_loader, device, class_names)
        
        # Log metrics to MLflow
        log_metrics_to_mlflow(metrics, class_names)
        
        # Print results
        print("\n" + "="*50)
        print(f"Test Accuracy: {metrics['accuracy']:.2f}%")
        print(f"Test Loss: {metrics['avg_loss']:.4f}")
        print(f"Test Precision: {metrics['precision']:.4f}")
        print(f"Test Recall: {metrics['recall']:.4f}")
        print(f"Test F1 Score: {metrics['f1']:.4f}")
        print("="*50)
        
        # Print classification report
        print("\nClassification Report:")
        for cls, vals in metrics['classification_report'].items():
            if isinstance(vals, dict):
                print(f"  Class {cls}:")
                print(f"    Precision: {vals['precision']:.4f}")
                print(f"    Recall: {vals['recall']:.4f}")
                print(f"    F1-score: {vals['f1-score']:.4f}")
                print(f"    Support: {vals['support']}")
        
        print("\nEvaluation results logged to MLflow.")


if __name__ == "__main__":
    main()