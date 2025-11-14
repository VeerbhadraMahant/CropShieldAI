"""
CropShield AI - Fully Automated Training Script
================================================

Complete self-contained training script that:
- Auto-detects GPU/CPU
- Auto-loads dataset with optimal settings
- Uses mixed-precision training (AMP)
- Implements early stopping
- Saves best model automatically
- Generates confusion matrix
- Reports final metrics

Usage:
    python train_auto.py                    # Train with defaults (25 epochs)
    python train_auto.py --epochs 50        # Custom epochs
    python train_auto.py --batch_size 64    # Custom batch size

No manual intervention required!
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from pathlib import Path
import time
from datetime import datetime
import json
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import argparse
import os
from typing import Tuple, Dict

# Import project modules
from models.model_factory import get_model
from fast_dataset import make_loaders


class AutoTrainer:
    """Fully automated trainer for CropShield AI"""
    
    def __init__(
        self,
        epochs: int = 25,
        batch_size: int = 32,
        learning_rate: float = 0.001,
        model_type: str = 'custom',
        early_stopping_patience: int = 10,
        save_dir: str = 'models'
    ):
        """
        Initialize automated trainer.
        
        Args:
            epochs: Number of training epochs
            batch_size: Batch size for training
            learning_rate: Initial learning rate
            model_type: Model architecture ('custom' or 'efficientnet')
            early_stopping_patience: Stop if no improvement for N epochs
            save_dir: Directory to save model
        """
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.model_type = model_type
        self.early_stopping_patience = early_stopping_patience
        self.save_dir = Path(save_dir)
        
        # Create save directory
        self.save_dir.mkdir(exist_ok=True)
        
        # Training history
        self.history = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': [],
            'learning_rates': []
        }
        
        # Best model tracking
        self.best_val_acc = 0.0
        self.best_epoch = 0
        self.patience_counter = 0
        
        # Device detection
        self.device = self._detect_device()
        
        # Mixed precision scaler
        self.scaler = GradScaler(enabled=torch.cuda.is_available())
        
        print("\n" + "="*80)
        print("🌾 CROPSHIELD AI - AUTOMATED TRAINING")
        print("="*80)
        print(f"📅 Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"🖥️  Device: {self.device}")
        print(f"📊 Epochs: {self.epochs}")
        print(f"📦 Batch Size: {self.batch_size}")
        print(f"🧠 Model: {self.model_type}")
        print(f"⚡ Mixed Precision: {torch.cuda.is_available()}")
        print(f"🛑 Early Stopping: {self.early_stopping_patience} epochs")
        print("="*80 + "\n")
    
    def _detect_device(self) -> torch.device:
        """Auto-detect best available device"""
        if torch.cuda.is_available():
            device = torch.device('cuda')
            gpu_name = torch.cuda.get_device_name(0)
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
            print(f"\n✅ GPU Detected: {gpu_name}")
            print(f"   Memory: {gpu_memory:.2f} GB")
        else:
            device = torch.device('cpu')
            print(f"\n⚠️  GPU Not Available - Using CPU")
        return device
    
    def load_data(self):
        """Auto-load and prepare datasets"""
        print("\n📂 Loading Dataset...")
        
        # Auto-detect num_workers based on CPU cores
        num_workers = 0 if os.name == 'nt' else min(8, os.cpu_count() - 1)
        
        # Load data with optimal settings
        result = make_loaders(
            data_dir='Database_resized',
            batch_size=self.batch_size,
            train_split=0.8,
            val_split=0.1,
            test_split=0.1,
            num_workers=num_workers,
            seed=42,
            augmentation_mode='moderate'
        )
        
        self.train_loader, self.val_loader, self.test_loader, self.class_names, self.dataset_info = result
        self.num_classes = len(self.class_names)
        
        print(f"✅ Dataset Loaded!")
        print(f"   Train: {len(self.train_loader.dataset)} images")
        print(f"   Val: {len(self.val_loader.dataset)} images")
        print(f"   Test: {len(self.test_loader.dataset)} images")
        print(f"   Classes: {self.num_classes}")
    
    def build_model(self):
        """Auto-build model with optimizer and scheduler"""
        print(f"\n🏗️  Building Model ({self.model_type})...")
        
        # Get model from factory
        self.model, self.optimizer, self.criterion, self.scheduler, _ = get_model(
            model_type=self.model_type,
            num_classes=self.num_classes,
            learning_rate=self.learning_rate
        )
        
        # Move to device
        self.model = self.model.to(self.device)
        
        # Count parameters
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        
        print(f"✅ Model Built!")
        print(f"   Total Parameters: {total_params:,}")
        print(f"   Trainable Parameters: {trainable_params:,}")
        print(f"   Model Size: {total_params * 4 / 1e6:.2f} MB")
    
    def train_epoch(self, epoch: int) -> Tuple[float, float]:
        """Train for one epoch"""
        self.model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        # Progress bar
        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch+1}/{self.epochs}")
        
        for batch_idx, (images, labels) in enumerate(pbar):
            images, labels = images.to(self.device), labels.to(self.device)
            
            # Zero gradients
            self.optimizer.zero_grad()
            
            # Mixed precision forward pass
            with autocast(enabled=torch.cuda.is_available()):
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
            
            # Backward pass with gradient scaling
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
            
            # Statistics
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            # Update progress bar
            accuracy = 100.0 * correct / total
            pbar.set_postfix({
                'Loss': f'{running_loss/(batch_idx+1):.4f}',
                'Acc': f'{accuracy:.2f}%'
            })
        
        epoch_loss = running_loss / len(self.train_loader)
        epoch_acc = 100.0 * correct / total
        
        return epoch_loss, epoch_acc
    
    def validate(self) -> Tuple[float, float]:
        """Validate model"""
        self.model.eval()
        running_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for images, labels in self.val_loader:
                images, labels = images.to(self.device), labels.to(self.device)
                
                # Forward pass
                with autocast(enabled=torch.cuda.is_available()):
                    outputs = self.model(images)
                    loss = self.criterion(outputs, labels)
                
                # Statistics
                running_loss += loss.item()
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
        
        val_loss = running_loss / len(self.val_loader)
        val_acc = 100.0 * correct / total
        
        return val_loss, val_acc
    
    def test(self) -> Dict:
        """Test model and generate confusion matrix"""
        print("\n🧪 Testing Model...")
        
        self.model.eval()
        all_preds = []
        all_labels = []
        correct = 0
        total = 0
        
        with torch.no_grad():
            for images, labels in tqdm(self.test_loader, desc="Testing"):
                images, labels = images.to(self.device), labels.to(self.device)
                
                with autocast(enabled=torch.cuda.is_available()):
                    outputs = self.model(images)
                
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
                
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        test_acc = 100.0 * correct / total
        
        # Confusion matrix
        cm = confusion_matrix(all_labels, all_preds)
        
        # Classification report
        report = classification_report(
            all_labels,
            all_preds,
            target_names=self.class_names,
            output_dict=True
        )
        
        print(f"✅ Test Accuracy: {test_acc:.2f}%")
        
        return {
            'accuracy': test_acc,
            'confusion_matrix': cm,
            'classification_report': report,
            'predictions': all_preds,
            'labels': all_labels
        }
    
    def plot_confusion_matrix(self, cm: np.ndarray, save_path: str):
        """Plot and save confusion matrix"""
        plt.figure(figsize=(12, 10))
        sns.heatmap(
            cm,
            annot=True,
            fmt='d',
            cmap='Blues',
            xticklabels=self.class_names,
            yticklabels=self.class_names,
            cbar_kws={'label': 'Count'}
        )
        plt.title('Confusion Matrix - CropShield AI')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"📊 Confusion matrix saved: {save_path}")
    
    def plot_training_history(self, save_path: str):
        """Plot training history"""
        fig, axes = plt.subplots(1, 2, figsize=(15, 5))
        
        # Loss plot
        axes[0].plot(self.history['train_loss'], label='Train Loss', marker='o')
        axes[0].plot(self.history['val_loss'], label='Val Loss', marker='s')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Loss')
        axes[0].set_title('Training and Validation Loss')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Accuracy plot
        axes[1].plot(self.history['train_acc'], label='Train Acc', marker='o')
        axes[1].plot(self.history['val_acc'], label='Val Acc', marker='s')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Accuracy (%)')
        axes[1].set_title('Training and Validation Accuracy')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"📈 Training history saved: {save_path}")
    
    def save_checkpoint(self, epoch: int, is_best: bool = False):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_val_acc': self.best_val_acc,
            'history': self.history,
            'num_classes': self.num_classes,
            'class_names': self.class_names,
            'model_type': self.model_type
        }
        
        if is_best:
            save_path = self.save_dir / 'cropshield_cnn.pth'
            torch.save(checkpoint, save_path)
            print(f"💾 Best model saved: {save_path}")
    
    def train(self):
        """Main training loop"""
        print("\n🚀 Starting Training...\n")
        
        start_time = time.time()
        
        for epoch in range(self.epochs):
            # Train epoch
            train_loss, train_acc = self.train_epoch(epoch)
            
            # Validate
            val_loss, val_acc = self.validate()
            
            # Update scheduler
            self.scheduler.step()
            current_lr = self.optimizer.param_groups[0]['lr']
            
            # Store history
            self.history['train_loss'].append(train_loss)
            self.history['train_acc'].append(train_acc)
            self.history['val_loss'].append(val_loss)
            self.history['val_acc'].append(val_acc)
            self.history['learning_rates'].append(current_lr)
            
            # Print epoch summary
            print(f"\n📊 Epoch {epoch+1}/{self.epochs} Summary:")
            print(f"   Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
            print(f"   Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%")
            print(f"   Learning Rate: {current_lr:.6f}")
            
            # Check for best model
            if val_acc > self.best_val_acc:
                self.best_val_acc = val_acc
                self.best_epoch = epoch
                self.patience_counter = 0
                self.save_checkpoint(epoch, is_best=True)
                print(f"   🎉 New best validation accuracy!")
            else:
                self.patience_counter += 1
                print(f"   No improvement ({self.patience_counter}/{self.early_stopping_patience})")
            
            # Early stopping
            if self.patience_counter >= self.early_stopping_patience:
                print(f"\n⏹️  Early stopping triggered after {epoch+1} epochs")
                print(f"   Best validation accuracy: {self.best_val_acc:.2f}% (Epoch {self.best_epoch+1})")
                break
            
            print("-" * 80)
        
        # Training complete
        training_time = time.time() - start_time
        hours = int(training_time // 3600)
        minutes = int((training_time % 3600) // 60)
        seconds = int(training_time % 60)
        
        print("\n" + "="*80)
        print("✅ TRAINING COMPLETE!")
        print("="*80)
        print(f"⏱️  Total Time: {hours:02d}:{minutes:02d}:{seconds:02d}")
        print(f"🎯 Best Validation Accuracy: {self.best_val_acc:.2f}% (Epoch {self.best_epoch+1})")
        print(f"💾 Model Saved: {self.save_dir / 'cropshield_cnn.pth'}")
        
        return self.history
    
    def run_complete_pipeline(self):
        """Execute complete training and evaluation pipeline"""
        try:
            # Load data
            self.load_data()
            
            # Build model
            self.build_model()
            
            # Train model
            self.train()
            
            # Load best model for testing
            print("\n📥 Loading best model for testing...")
            checkpoint = torch.load(self.save_dir / 'cropshield_cnn.pth')
            self.model.load_state_dict(checkpoint['model_state_dict'])
            
            # Test model
            test_results = self.test()
            
            # Generate confusion matrix
            cm_path = self.save_dir / 'confusion_matrix.png'
            self.plot_confusion_matrix(test_results['confusion_matrix'], str(cm_path))
            
            # Plot training history
            history_path = self.save_dir / 'training_history.png'
            self.plot_training_history(str(history_path))
            
            # Save final results
            results = {
                'best_val_acc': self.best_val_acc,
                'best_epoch': self.best_epoch + 1,
                'test_acc': test_results['accuracy'],
                'training_history': self.history,
                'classification_report': test_results['classification_report']
            }
            
            results_path = self.save_dir / 'training_results.json'
            with open(results_path, 'w') as f:
                json.dump(results, f, indent=2)
            
            print("\n" + "="*80)
            print("🎉 ALL DONE!")
            print("="*80)
            print(f"📊 Final Results:")
            print(f"   Best Validation Accuracy: {self.best_val_acc:.2f}%")
            print(f"   Test Accuracy: {test_results['accuracy']:.2f}%")
            print(f"\n📁 Output Files:")
            print(f"   Model: {self.save_dir / 'cropshield_cnn.pth'}")
            print(f"   Confusion Matrix: {cm_path}")
            print(f"   Training History: {history_path}")
            print(f"   Results JSON: {results_path}")
            print("="*80 + "\n")
            
            return results
            
        except Exception as e:
            print(f"\n❌ Error during training: {e}")
            import traceback
            traceback.print_exc()
            raise


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description='CropShield AI - Automated Training')
    parser.add_argument('--epochs', type=int, default=25, help='Number of epochs (default: 25)')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size (default: 32)')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate (default: 0.001)')
    parser.add_argument('--model_type', type=str, default='custom', choices=['custom', 'efficientnet'], 
                        help='Model architecture (default: custom)')
    parser.add_argument('--early_stopping_patience', type=int, default=10, 
                        help='Early stopping patience (default: 10)')
    parser.add_argument('--save_dir', type=str, default='models', help='Save directory (default: models)')
    
    args = parser.parse_args()
    
    # Create trainer
    trainer = AutoTrainer(
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        model_type=args.model_type,
        early_stopping_patience=args.early_stopping_patience,
        save_dir=args.save_dir
    )
    
    # Run complete pipeline
    trainer.run_complete_pipeline()


if __name__ == '__main__':
    main()
