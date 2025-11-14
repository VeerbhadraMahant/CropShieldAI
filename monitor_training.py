"""
Monitor Training Progress
========================

Quick script to check training status without interrupting it.

Usage:
    python monitor_training.py
"""

import os
import json
from pathlib import Path
import time

def monitor_training():
    """Monitor training progress"""
    models_dir = Path('models')
    
    print("\n" + "="*80)
    print("🔍 TRAINING MONITOR")
    print("="*80)
    
    # Check for model file
    model_path = models_dir / 'cropshield_cnn.pth'
    if model_path.exists():
        model_time = time.ctime(os.path.getmtime(model_path))
        model_size = os.path.getsize(model_path) / (1024**2)
        print(f"✅ Model Found: {model_path}")
        print(f"   Last Modified: {model_time}")
        print(f"   Size: {model_size:.2f} MB")
    else:
        print(f"⏳ Model Not Yet Saved: {model_path}")
    
    # Check for results
    results_path = models_dir / 'training_results.json'
    if results_path.exists():
        with open(results_path) as f:
            results = json.load(f)
        
        print(f"\n✅ Training Results Found:")
        print(f"   Best Validation Acc: {results['best_val_acc']:.2f}%")
        print(f"   Best Epoch: {results['best_epoch']}")
        print(f"   Test Accuracy: {results['test_acc']:.2f}%")
        
        # Show last few epochs
        history = results['training_history']
        num_epochs = len(history['train_loss'])
        print(f"\n📊 Training History ({num_epochs} epochs):")
        for i in range(max(0, num_epochs-5), num_epochs):
            print(f"   Epoch {i+1}: Train Acc={history['train_acc'][i]:.2f}%, Val Acc={history['val_acc'][i]:.2f}%")
    else:
        print(f"\n⏳ Results Not Yet Available: {results_path}")
    
    # Check for confusion matrix
    cm_path = models_dir / 'confusion_matrix.png'
    if cm_path.exists():
        print(f"\n✅ Confusion Matrix: {cm_path}")
    else:
        print(f"\n⏳ Confusion Matrix Not Yet Generated: {cm_path}")
    
    # Check for training history plot
    history_path = models_dir / 'training_history.png'
    if history_path.exists():
        print(f"✅ Training History Plot: {history_path}")
    else:
        print(f"⏳ Training History Plot Not Yet Generated: {history_path}")
    
    print("\n" + "="*80)
    print("💡 Tip: Run this script again to see updated progress!")
    print("="*80 + "\n")

if __name__ == '__main__':
    monitor_training()
