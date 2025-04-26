#!/usr/bin/env python3

"""
Test script to evaluate the retrained model on test data.
This script loads a trained model and evaluates its performance on the test dataset,
logging results to MLflow.
"""

from pathlib import Path
import sys
import os
import argparse
import torch
import torch.nn as nn
import numpy as np
import mlflow
import mlflow.pytorch
import json
import io
import logging
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_fscore_support
from torch.utils.data import DataLoader

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# Add required paths
model_building_path = Path(__file__).parent.parent / "model_building_pipeline"
drift_pipeline_path = Path(__file__).parent.parent / "drift_pipeline"

for p in [model_building_path, drift_pipeline_path]:
    if str(p) not in sys.path:
        sys.path.append(str(p))

# Import required modules
from drift_pipeline.database import DriftDatabase
from prepare_retrain_data import prepare_retraining_data
from model_building_pipeline.model import ResNetModel

def update_training_status():
    """Mark drifted images as used in training.
    
    Returns:
        int: Number of images updated
    """
    db = DriftDatabase()
    updated_count = 0
    
    try:
        if not db.connect():
            logger.error("Failed to connect to the database.")
            return 0
        
        # Update all drifted images not yet used in training
        result = db.collection.update_many(
            {'drift_detected': True, 'used_in_training': False},
            {'$set': {'used_in_training': True}}
        )
        updated_count = result.modified_count
        
        logger.info(f"Updated training status for {updated_count} drifted images")
    except Exception as e:
        logger.error(f"Error updating training status: {e}")
    finally:
        db.close()
    
    return updated_count


def parse_args():
    """Parse command line arguments for model testing."""
    parser = argparse.ArgumentParser(description='Test the trained ResNet model')
    
    parser.add_argument('--model-path', type=str, default='best_model_resnet.pth',
                        help='Path to the saved model weights')
    parser.add_argument('--experiment-name', type=str, default='resnet_testing',
                        help='MLflow experiment name')
    parser.add_argument('--tracking-uri', type=str, default=None,
                        help='MLflow tracking URI')
    parser.add_argument('-b', '--batch-size', type=int, default=32,
                        help='Testing batch size')
    parser.add_argument('--num-classes', type=int, default=10,
                        help='Number of classes in the dataset')
    parser.add_argument('--dropout-rate', type=float, default=0.5,
                        help='Dropout rate used in the model')
    parser.add_argument('--load-datasets', type=str,
                      help='Load prepared datasets from pickle file')
    parser.add_argument('--update-db-status', action='store_true',
                      help='Update database to mark drifted images as used in training')
    
    return parser.parse_args()


def evaluate_model(model, test_loader, device, class_names=None):
    """Evaluate the model on test data."""
    model.eval()
    criterion = nn.CrossEntropyLoss()
    
    test_loss = 0.0
    correct = 0
    total = 0
    
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
            
            all_preds.extend(predicted.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())
    
    # Calculate metrics
    accuracy = 100.0 * correct / total
    avg_loss = test_loss / len(test_loader)
    precision, recall, f1, _ = precision_recall_fscore_support(
        all_targets, all_preds, average='weighted'
    )
    
    # Create classification report
    report = classification_report(
        all_targets, all_preds, 
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
        'confusion_matrix': conf_matrix
    }


def plot_confusion_matrix(conf_matrix, class_names):
    """Create confusion matrix plot."""
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
    """Log evaluation metrics to MLflow."""
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
                f'class_{cls_name}_f1': vals['f1-score']
            })
    
    # Create and log confusion matrix image
    if class_names:
        fig = plot_confusion_matrix(metrics['confusion_matrix'], class_names)
        mlflow.log_figure(fig, "confusion_matrix.png")
        plt.close(fig)


def main():
    """Main function to test the model."""
    args = parse_args()
    
    # Set MLflow tracking
    if args.tracking_uri:
        mlflow.set_tracking_uri(args.tracking_uri)
    
    mlflow.set_experiment(args.experiment_name)
    
    # Start MLflow run
    with mlflow.start_run():
        # Log parameters
        mlflow.log_params({
            'model_path': args.model_path,
            'num_classes': args.num_classes,
            'dropout_rate': args.dropout_rate
        })
        
        # Set device for testing
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Using device: {device}")
        mlflow.log_param('device', device.type)
        
        # Load datasets
        if args.load_datasets:
            import pickle
            with open(args.load_datasets, 'rb') as f:
                datasets = pickle.load(f)
            
            # Create test loader from saved dataset
            test_dataset = datasets['test']
            test_loader = DataLoader(
                test_dataset,
                batch_size=args.batch_size,
                shuffle=False,
                num_workers=4,
                pin_memory=True
            )
            logger.info(f"Loaded test dataset with {len(test_dataset)} samples")
        else:
            logger.error("No dataset provided. Use --load-datasets to specify a dataset.")
            return
        
        # Initialize and load model
        model = ResNetModel(
            num_classes=args.num_classes,
            dropout_rate=args.dropout_rate
        )
        
        if os.path.exists(args.model_path):
            model.load_state_dict(torch.load(args.model_path, map_location=device))
            logger.info(f"Model weights loaded from {args.model_path}")
        else:
            logger.error(f"Model weights not found at {args.model_path}")
            return
        
        model = model.to(device)
        
        # Get class names
        class_names = None
        if 'classes' in datasets:
            class_names = datasets['classes']
            logger.info(f"Found {len(class_names)} classes in dataset")
        
        # Evaluate the model
        logger.info("Evaluating model on test dataset...")
        metrics = evaluate_model(model, test_loader, device, class_names)
        
        # Log metrics to MLflow
        log_metrics_to_mlflow(metrics, class_names)
        
        # Print results
        logger.info("="*50)
        logger.info(f"Test Accuracy: {metrics['accuracy']:.2f}%")
        logger.info(f"Test Precision: {metrics['precision']:.4f}")
        logger.info(f"Test Recall: {metrics['recall']:.4f}")
        logger.info(f"Test F1 Score: {metrics['f1']:.4f}")
        logger.info("="*50)
        
        # Update database if requested
        if args.update_db_status:
            logger.info("Updating database to mark drifted images as used in training...")
            updated_count = update_training_status()
            logger.info(f"Updated {updated_count} images in the database")
            mlflow.log_param("images_marked_as_used", updated_count)


if __name__ == "__main__":
    main()