"""
CropShield AI - Model Evaluation Script

Comprehensive evaluation script for trained models.
Computes detailed metrics, confusion matrix, and per-class performance.

Usage:
    python evaluate.py --checkpoint checkpoints/best.pth --model custom
    python evaluate.py --checkpoint checkpoints/efficientnet_best.pth --model efficientnet_b0
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import autocast
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, 
    f1_score, 
    precision_score, 
    recall_score,
    confusion_matrix,
    classification_report
)
import pandas as pd
import json
import argparse
from pathlib import Path
from tqdm import tqdm
import time
from typing import Dict, List, Tuple

# Import local modules
from fast_dataset import make_loaders
from models.model_factory import get_model


class ModelEvaluator:
    """
    Comprehensive model evaluation class.
    
    Evaluates model on test set with:
    - Overall metrics (accuracy, macro F1)
    - Per-class metrics (precision, recall, F1, support)
    - Confusion matrix visualization
    - Detailed JSON report
    """
    
    def __init__(
        self,
        model: nn.Module,
        test_loader: torch.utils.data.DataLoader,
        class_names: List[str],
        device: torch.device,
        use_amp: bool = True,
        results_dir: str = 'results'
    ):
        """
        Initialize evaluator.
        
        Args:
            model: Trained PyTorch model
            test_loader: Test data loader
            class_names: List of class names
            device: Device to run inference on
            use_amp: Use mixed precision inference
            results_dir: Directory to save results
        """
        self.model = model
        self.test_loader = test_loader
        self.class_names = class_names
        self.device = device
        self.use_amp = use_amp and device.type == 'cuda'
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(exist_ok=True, parents=True)
        
        # Set model to eval mode
        self.model.eval()
        
        print(f"\n{'='*60}")
        print("🔍 MODEL EVALUATOR INITIALIZED")
        print(f"{'='*60}")
        print(f"Device:              {device}")
        print(f"Mixed precision:     {self.use_amp}")
        print(f"Test samples:        {len(test_loader.dataset)}")
        print(f"Batch size:          {test_loader.batch_size}")
        print(f"Number of classes:   {len(class_names)}")
        print(f"Results directory:   {self.results_dir}")
        print(f"{'='*60}\n")
    
    @torch.no_grad()
    def collect_predictions(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Collect predictions on test set with efficient memory usage.
        
        Returns:
            Tuple of (all_labels, all_predictions, all_probabilities)
        """
        print("📊 Collecting predictions on test set...")
        
        all_labels = []
        all_predictions = []
        all_probabilities = []
        
        start_time = time.time()
        
        # Progress bar
        pbar = tqdm(
            self.test_loader,
            desc="Evaluating",
            unit="batch",
            ncols=100
        )
        
        for images, labels in pbar:
            # Move to device
            images = images.to(self.device, non_blocking=True)
            labels = labels.to(self.device, non_blocking=True)
            
            # Forward pass with optional AMP
            if self.use_amp:
                with autocast():
                    outputs = self.model(images)
            else:
                outputs = self.model(images)
            
            # Get probabilities (softmax)
            probabilities = F.softmax(outputs, dim=1)
            
            # Get predictions (argmax)
            predictions = torch.argmax(outputs, dim=1)
            
            # Move to CPU and convert to numpy (avoid GPU memory accumulation)
            all_labels.append(labels.cpu().numpy())
            all_predictions.append(predictions.cpu().numpy())
            all_probabilities.append(probabilities.cpu().numpy())
        
        # Concatenate all batches
        all_labels = np.concatenate(all_labels)
        all_predictions = np.concatenate(all_predictions)
        all_probabilities = np.concatenate(all_probabilities)
        
        elapsed = time.time() - start_time
        throughput = len(all_labels) / elapsed
        
        print(f"✅ Predictions collected: {len(all_labels)} samples in {elapsed:.1f}s ({throughput:.1f} img/s)")
        
        return all_labels, all_predictions, all_probabilities
    
    def compute_metrics(
        self,
        labels: np.ndarray,
        predictions: np.ndarray,
        probabilities: np.ndarray
    ) -> Dict:
        """
        Compute comprehensive evaluation metrics.
        
        Args:
            labels: True labels
            predictions: Predicted labels
            probabilities: Predicted probabilities
            
        Returns:
            Dictionary containing all metrics
        """
        print("\n📈 Computing metrics...")
        
        # Overall metrics
        accuracy = accuracy_score(labels, predictions)
        macro_f1 = f1_score(labels, predictions, average='macro', zero_division=0)
        weighted_f1 = f1_score(labels, predictions, average='weighted', zero_division=0)
        
        # Per-class metrics
        precision_per_class = precision_score(
            labels, predictions, average=None, zero_division=0
        )
        recall_per_class = recall_score(
            labels, predictions, average=None, zero_division=0
        )
        f1_per_class = f1_score(
            labels, predictions, average=None, zero_division=0
        )
        
        # Support (number of samples per class)
        unique, counts = np.unique(labels, return_counts=True)
        support_per_class = np.zeros(len(self.class_names))
        support_per_class[unique] = counts
        
        # Confusion matrix
        cm = confusion_matrix(labels, predictions)
        
        # Build metrics dictionary
        metrics = {
            'overall': {
                'accuracy': float(accuracy),
                'macro_f1': float(macro_f1),
                'weighted_f1': float(weighted_f1),
                'num_samples': int(len(labels)),
                'num_classes': int(len(self.class_names))
            },
            'per_class': {}
        }
        
        # Add per-class metrics
        for i, class_name in enumerate(self.class_names):
            metrics['per_class'][class_name] = {
                'precision': float(precision_per_class[i]),
                'recall': float(recall_per_class[i]),
                'f1_score': float(f1_per_class[i]),
                'support': int(support_per_class[i])
            }
        
        # Print summary
        print(f"\n{'='*60}")
        print("📊 EVALUATION RESULTS")
        print(f"{'='*60}")
        print(f"Overall Accuracy:    {accuracy:.4f} ({accuracy*100:.2f}%)")
        print(f"Macro F1:            {macro_f1:.4f}")
        print(f"Weighted F1:         {weighted_f1:.4f}")
        print(f"Test samples:        {len(labels)}")
        print(f"{'='*60}\n")
        
        return metrics, cm
    
    def save_confusion_matrix(
        self,
        cm: np.ndarray,
        filename: str = 'confusion_matrix.png'
    ):
        """
        Create and save confusion matrix visualization.
        
        Args:
            cm: Confusion matrix
            filename: Output filename
        """
        print(f"📊 Creating confusion matrix visualization...")
        
        # Create figure
        fig, ax = plt.subplots(figsize=(14, 12))
        
        # Normalize confusion matrix (percentage)
        cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        
        # Create heatmap
        sns.heatmap(
            cm_normalized,
            annot=True,
            fmt='.2%',
            cmap='Blues',
            xticklabels=self.class_names,
            yticklabels=self.class_names,
            cbar_kws={'label': 'Percentage'},
            ax=ax
        )
        
        # Labels and title
        ax.set_xlabel('Predicted Label', fontsize=12, fontweight='bold')
        ax.set_ylabel('True Label', fontsize=12, fontweight='bold')
        ax.set_title(
            'Confusion Matrix - Test Set Evaluation',
            fontsize=14,
            fontweight='bold',
            pad=20
        )
        
        # Rotate x-axis labels for readability
        plt.setp(ax.get_xticklabels(), rotation=45, ha='right', rotation_mode='anchor')
        plt.setp(ax.get_yticklabels(), rotation=0)
        
        # Tight layout
        plt.tight_layout()
        
        # Save
        output_path = self.results_dir / filename
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✅ Confusion matrix saved: {output_path}")
    
    def save_class_report_csv(
        self,
        metrics: Dict,
        filename: str = 'class_report.csv'
    ):
        """
        Save per-class metrics to CSV.
        
        Args:
            metrics: Metrics dictionary
            filename: Output filename
        """
        print(f"📄 Creating per-class metrics CSV...")
        
        # Build DataFrame
        data = []
        for class_name, class_metrics in metrics['per_class'].items():
            data.append({
                'class_name': class_name,
                'precision': class_metrics['precision'],
                'recall': class_metrics['recall'],
                'f1_score': class_metrics['f1_score'],
                'support': class_metrics['support']
            })
        
        df = pd.DataFrame(data)
        
        # Sort by F1 score (descending)
        df = df.sort_values('f1_score', ascending=False)
        
        # Save to CSV
        output_path = self.results_dir / filename
        df.to_csv(output_path, index=False, float_format='%.4f')
        
        print(f"✅ Class report saved: {output_path}")
        
        # Print top 5 and bottom 5
        print(f"\n{'='*60}")
        print("🏆 TOP 5 CLASSES (by F1 Score)")
        print(f"{'='*60}")
        for _, row in df.head(5).iterrows():
            print(f"{row['class_name']:40s} | F1: {row['f1_score']:.4f} | Precision: {row['precision']:.4f} | Recall: {row['recall']:.4f}")
        
        print(f"\n{'='*60}")
        print("⚠️  BOTTOM 5 CLASSES (by F1 Score)")
        print(f"{'='*60}")
        for _, row in df.tail(5).iterrows():
            print(f"{row['class_name']:40s} | F1: {row['f1_score']:.4f} | Precision: {row['precision']:.4f} | Recall: {row['recall']:.4f}")
        print(f"{'='*60}\n")
    
    def save_metrics_json(
        self,
        metrics: Dict,
        filename: str = 'test_metrics.json'
    ):
        """
        Save all metrics to JSON file.
        
        Args:
            metrics: Metrics dictionary
            filename: Output filename
        """
        print(f"💾 Saving metrics to JSON...")
        
        output_path = self.results_dir / filename
        
        with open(output_path, 'w') as f:
            json.dump(metrics, f, indent=2)
        
        print(f"✅ Metrics saved: {output_path}")
    
    def evaluate(self) -> Dict:
        """
        Run complete evaluation pipeline.
        
        Returns:
            Dictionary containing all metrics
        """
        print(f"\n{'='*60}")
        print("🚀 STARTING EVALUATION")
        print(f"{'='*60}\n")
        
        start_time = time.time()
        
        # 1. Collect predictions
        labels, predictions, probabilities = self.collect_predictions()
        
        # 2. Compute metrics
        metrics, cm = self.compute_metrics(labels, predictions, probabilities)
        
        # 3. Save confusion matrix
        self.save_confusion_matrix(cm)
        
        # 4. Save class report CSV
        self.save_class_report_csv(metrics)
        
        # 5. Save metrics JSON
        self.save_metrics_json(metrics)
        
        elapsed = time.time() - start_time
        
        print(f"\n{'='*60}")
        print("✅ EVALUATION COMPLETE")
        print(f"{'='*60}")
        print(f"Total time:          {elapsed:.1f}s")
        print(f"Results saved to:    {self.results_dir}")
        print(f"{'='*60}\n")
        
        return metrics


def load_checkpoint(
    checkpoint_path: str,
    model: nn.Module,
    device: torch.device
) -> nn.Module:
    """
    Load model from checkpoint.
    
    Args:
        checkpoint_path: Path to checkpoint file
        model: Model to load weights into
        device: Device to map checkpoint to
        
    Returns:
        Model with loaded weights
    """
    checkpoint_path = Path(checkpoint_path)
    
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    
    print(f"📂 Loading checkpoint: {checkpoint_path}")
    
    try:
        checkpoint = torch.load(checkpoint_path, map_location=device)
        
        # Load model state
        model.load_state_dict(checkpoint['model_state_dict'])
        
        print(f"✅ Model loaded successfully")
        
        # Print checkpoint info
        if 'epoch' in checkpoint:
            print(f"   Epoch: {checkpoint['epoch']}")
        if 'best_val_loss' in checkpoint:
            print(f"   Best val loss: {checkpoint['best_val_loss']:.4f}")
        if 'best_val_acc' in checkpoint:
            print(f"   Best val acc: {checkpoint['best_val_acc']:.2f}%")
        
        return model
        
    except Exception as e:
        raise RuntimeError(f"Failed to load checkpoint: {e}")


def main(args):
    """Main evaluation function."""
    
    print(f"\n{'='*60}")
    print("🌾 CROPSHIELD AI - MODEL EVALUATION")
    print(f"{'='*60}")
    print(f"Model:               {args.model}")
    print(f"Checkpoint:          {args.checkpoint}")
    print(f"Batch size:          {args.batch_size}")
    print(f"Mixed precision:     {args.amp}")
    print(f"Results directory:   {args.results_dir}")
    print(f"{'='*60}\n")
    
    # 1. Load data
    print("📦 Loading test dataset...")
    _, _, test_loader, class_names, info = make_loaders(
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        augmentation_mode='none'  # No augmentation for evaluation
    )
    
    num_classes = len(class_names)
    print(f"✅ Dataset loaded: {num_classes} classes, {len(test_loader.dataset)} test samples")
    
    # 2. Create model
    print(f"\n📦 Creating model: {args.model}")
    model, _, _, _, device = get_model(
        model_type=args.model,
        num_classes=num_classes,
        verbose=True
    )
    
    # 3. Load checkpoint
    print(f"\n📂 Loading trained model...")
    model = load_checkpoint(args.checkpoint, model, device)
    
    # 4. Create evaluator
    evaluator = ModelEvaluator(
        model=model,
        test_loader=test_loader,
        class_names=class_names,
        device=device,
        use_amp=args.amp,
        results_dir=args.results_dir
    )
    
    # 5. Run evaluation
    metrics = evaluator.evaluate()
    
    # 6. Print final summary
    print(f"\n{'='*60}")
    print("📊 FINAL SUMMARY")
    print(f"{'='*60}")
    print(f"Overall Accuracy:    {metrics['overall']['accuracy']:.4f} ({metrics['overall']['accuracy']*100:.2f}%)")
    print(f"Macro F1 Score:      {metrics['overall']['macro_f1']:.4f}")
    print(f"Weighted F1 Score:   {metrics['overall']['weighted_f1']:.4f}")
    print(f"\nResults saved to:")
    print(f"  📊 {args.results_dir}/confusion_matrix.png")
    print(f"  📄 {args.results_dir}/class_report.csv")
    print(f"  💾 {args.results_dir}/test_metrics.json")
    print(f"{'='*60}\n")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Evaluate trained CropShield AI model on test set'
    )
    
    # Model configuration
    parser.add_argument(
        '--model',
        type=str,
        required=True,
        choices=['custom', 'efficientnet', 'efficientnet_b0'],
        help='Model architecture to evaluate'
    )
    parser.add_argument(
        '--checkpoint',
        type=str,
        required=True,
        help='Path to trained model checkpoint'
    )
    
    # Data configuration
    parser.add_argument(
        '--batch_size',
        type=int,
        default=64,
        help='Batch size for evaluation (default: 64)'
    )
    parser.add_argument(
        '--num_workers',
        type=int,
        default=8,
        help='Number of data loading workers (default: 8)'
    )
    
    # Inference configuration
    parser.add_argument(
        '--amp',
        type=bool,
        default=True,
        help='Use mixed precision inference (default: True)'
    )
    
    # Output configuration
    parser.add_argument(
        '--results_dir',
        type=str,
        default='results',
        help='Directory to save evaluation results (default: results)'
    )
    
    args = parser.parse_args()
    
    # Run evaluation
    main(args)
