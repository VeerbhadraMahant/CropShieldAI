"""
CropShield AI - Quick Model Evaluation
========================================

Automatic evaluation script that works with trained cropshield_cnn.pth model.
No arguments required - fully automatic!

Usage:
    python quick_evaluate.py

Outputs:
    - Console summary with test accuracy
    - results/confusion_matrix.png
    - results/test_metrics.json
"""

import torch
import torch.nn as nn
from pathlib import Path
import json
import numpy as np
from sklearn.metrics import (
    confusion_matrix, 
    classification_report,
    accuracy_score,
    precision_recall_fscore_support
)
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from datetime import datetime
import os

# Import project modules
from models.model_factory import get_model
from fast_dataset import make_loaders


def evaluate_model():
    """Complete automatic evaluation"""
    
    print("\n" + "="*80)
    print("📊 CROPSHIELD AI - MODEL EVALUATION")
    print("="*80)
    
    # Configuration
    model_path = Path('models/cropshield_cnn.pth')
    results_dir = Path('results')
    batch_size = 32
    
    # Create results directory
    results_dir.mkdir(exist_ok=True)
    
    # Device detection
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🖥️  Device: {device}")
    if torch.cuda.is_available():
        print(f"   GPU: {torch.cuda.get_device_name(0)}")
    
    # Step 1: Load Model
    print("\n📥 Loading Model...")
    if not model_path.exists():
        print(f"❌ Model not found: {model_path}")
        print("   Please train the model first: python train_auto.py")
        return
    
    checkpoint = torch.load(model_path, map_location=device)
    num_classes = checkpoint['num_classes']
    class_names = checkpoint['class_names']
    model_type = checkpoint.get('model_type', 'custom')
    
    print(f"✅ Model Loaded:")
    print(f"   Type: {model_type}")
    print(f"   Classes: {num_classes}")
    print(f"   Training Epoch: {checkpoint['epoch'] + 1}")
    print(f"   Best Val Acc: {checkpoint['best_val_acc']:.2f}%")
    
    # Create model and load weights
    model, _, _, _, _ = get_model(
        model_type=model_type,
        num_classes=num_classes,
        learning_rate=0.001
    )
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    # Step 2: Load Test Data
    print("\n📂 Loading Test Dataset...")
    num_workers = 0 if os.name == 'nt' else min(8, os.cpu_count() - 1)
    
    _, _, test_loader, _, _ = make_loaders(
        data_dir='Database_resized',
        batch_size=batch_size,
        train_split=0.8,
        val_split=0.1,
        test_split=0.1,
        num_workers=num_workers,
        seed=42,
        augmentation_mode='moderate'
    )
    
    print(f"✅ Test Dataset: {len(test_loader.dataset)} images")
    
    # Step 3: Evaluate
    print("\n🧪 Evaluating on Test Set...")
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for images, labels in tqdm(test_loader, desc="Testing"):
            images = images.to(device)
            labels = labels.to(device)
            
            outputs = model(images)
            _, predicted = outputs.max(1)
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    predictions = np.array(all_preds)
    labels = np.array(all_labels)
    
    # Step 4: Compute Metrics
    print("\n📈 Computing Metrics...")
    
    # Overall accuracy
    accuracy = accuracy_score(labels, predictions) * 100
    
    # Per-class metrics
    precision, recall, f1, support = precision_recall_fscore_support(
        labels, predictions, average=None, zero_division=0
    )
    
    # Macro averages
    precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
        labels, predictions, average='macro', zero_division=0
    )
    
    # Weighted averages
    precision_weighted, recall_weighted, f1_weighted, _ = precision_recall_fscore_support(
        labels, predictions, average='weighted', zero_division=0
    )
    
    # Confusion matrix
    cm = confusion_matrix(labels, predictions)
    
    # Classification report
    report_dict = classification_report(
        labels, predictions, target_names=class_names,
        output_dict=True, zero_division=0
    )
    
    # Step 5: Save Confusion Matrix
    print("\n📊 Generating Confusion Matrix...")
    
    plt.figure(figsize=(16, 14))
    sns.heatmap(
        cm, annot=True, fmt='d', cmap='Blues',
        xticklabels=class_names, yticklabels=class_names,
        cbar_kws={'label': 'Count'}, linewidths=0.5, linecolor='gray'
    )
    plt.title(
        f'Confusion Matrix - CropShield AI\nTest Accuracy: {accuracy:.2f}%',
        fontsize=16, fontweight='bold', pad=20
    )
    plt.ylabel('True Label', fontsize=12, fontweight='bold')
    plt.xlabel('Predicted Label', fontsize=12, fontweight='bold')
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    
    cm_path = results_dir / 'confusion_matrix.png'
    plt.savefig(cm_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✅ Saved: {cm_path}")
    
    # Step 6: Save Metrics JSON
    print("\n💾 Saving Metrics...")
    
    metrics = {
        'test_accuracy': float(accuracy),
        'precision_macro': float(precision_macro * 100),
        'recall_macro': float(recall_macro * 100),
        'f1_macro': float(f1_macro * 100),
        'precision_weighted': float(precision_weighted * 100),
        'recall_weighted': float(recall_weighted * 100),
        'f1_weighted': float(f1_weighted * 100),
        'per_class_metrics': {},
        'classification_report': report_dict,
        'confusion_matrix': cm.tolist(),
        'total_samples': int(len(labels)),
        'num_classes': int(num_classes),
        'class_names': class_names,
        'metadata': {
            'model_path': str(model_path),
            'evaluation_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'device': str(device)
        }
    }
    
    # Per-class metrics
    for i, class_name in enumerate(class_names):
        metrics['per_class_metrics'][class_name] = {
            'precision': float(precision[i] * 100),
            'recall': float(recall[i] * 100),
            'f1_score': float(f1[i] * 100),
            'support': int(support[i])
        }
    
    json_path = results_dir / 'test_metrics.json'
    with open(json_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"✅ Saved: {json_path}")
    
    # Step 7: Print Summary
    print("\n" + "="*80)
    print("✅ EVALUATION COMPLETE!")
    print("="*80)
    print(f"\n📊 RESULTS:")
    print(f"   ✅ Test Accuracy:  {accuracy:.1f}%")
    print(f"   Precision (Macro): {precision_macro*100:.2f}%")
    print(f"   Recall (Macro):    {recall_macro*100:.2f}%")
    print(f"   F1-Score (Macro):  {f1_macro*100:.2f}%")
    
    # Top 5 classes
    print(f"\n🏆 TOP 5 CLASSES (by F1-Score):")
    sorted_classes = sorted(
        metrics['per_class_metrics'].items(),
        key=lambda x: x[1]['f1_score'],
        reverse=True
    )
    for i, (cls, m) in enumerate(sorted_classes[:5], 1):
        print(f"   {i}. {cls}: {m['f1_score']:.1f}% F1")
    
    print(f"\n📁 OUTPUT FILES:")
    print(f"   {cm_path}")
    print(f"   {json_path}")
    
    print("\n" + "="*80)
    print(f"✅ Test Accuracy: {accuracy:.1f}%")
    print("="*80 + "\n")
    
    return metrics


if __name__ == '__main__':
    evaluate_model()
