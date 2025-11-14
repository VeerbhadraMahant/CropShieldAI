"""
CropShield AI - Training Script
================================

Production-ready training loop with mixed-precision training, gradient accumulation,
progress tracking, and comprehensive logging.

Optimized for RTX 4060 Laptop GPU with PyTorch AMP for faster training.

Features:
- Mixed-precision training (torch.cuda.amp)
- Gradient accumulation for effective larger batch sizes
- Progress bars (tqdm)
- Model checkpointing (best model saving)
- Training history logging (JSON)
- Transfer learning support (backbone freezing)
- Early stopping (optional)
- Learning rate scheduling

Usage:
    # Train custom CNN
    python train.py --model custom --epochs 25
    
    # Train EfficientNet with transfer learning
    python train.py --model efficientnet --epochs 25 --freeze_backbone
    
    # With gradient accumulation
    python train.py --model custom --epochs 25 --accumulation_steps 2
    
    # Resume from checkpoint
    python train.py --model custom --epochs 25 --resume models/checkpoint.pth
"""

import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from typing import Dict, Tuple, Optional, Any
import json
import argparse
from pathlib import Path
from tqdm import tqdm
import time
from datetime import datetime
import sys
import os
import shutil

# Import project modules
from models import get_model
from fast_dataset import make_loaders


class Trainer:
    """
    Training manager for CropShield AI models.
    
    Handles training loop, validation, checkpointing, and logging.
    """
    
    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        criterion: nn.Module,
        scheduler: torch.optim.lr_scheduler._LRScheduler,
        device: torch.device,
        train_loader: DataLoader,
        val_loader: DataLoader,
        test_loader: DataLoader,
        num_classes: int = 22,
        accumulation_steps: int = 1,
        use_amp: bool = True,
        clip_grad_norm: float = None,
        checkpoint_dir: str = 'models',
        log_dir: str = '.',
        model_name: str = 'cropshield_cnn'
    ):
        """
        Initialize trainer.
        
        Args:
            model: PyTorch model
            optimizer: Optimizer
            criterion: Loss function
            scheduler: Learning rate scheduler
            device: Device (cuda/cpu)
            train_loader: Training data loader
            val_loader: Validation data loader
            test_loader: Test data loader
            num_classes: Number of classes
            accumulation_steps: Gradient accumulation steps
            use_amp: Use mixed-precision training
            clip_grad_norm: Gradient clipping max norm (None to disable)
            checkpoint_dir: Directory to save checkpoints
            log_dir: Directory to save logs
            model_name: Name for saving model
        """
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.scheduler = scheduler
        self.device = device
        
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        
        self.num_classes = num_classes
        self.accumulation_steps = accumulation_steps
        self.use_amp = use_amp and torch.cuda.is_available()
        self.clip_grad_norm = clip_grad_norm
        
        # Mixed precision scaler
        self.scaler = GradScaler() if self.use_amp else None
        
        # Paths
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.log_dir = Path(log_dir)
        self.model_name = model_name
        
        # Training state
        self.current_epoch = 0
        self.best_val_loss = float('inf')
        self.best_val_acc = 0.0
        self.epochs_without_improvement = 0
        self.history = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': [],
            'learning_rate': []
        }
        
        # Dataset info
        self.train_size = len(train_loader.dataset)
        self.val_size = len(val_loader.dataset)
        self.test_size = len(test_loader.dataset)
        
        print(f"\n{'='*60}")
        print("🚀 TRAINER INITIALIZED")
        print(f"{'='*60}")
        print(f"Device:              {device}")
        print(f"Mixed Precision:     {self.use_amp}")
        print(f"Accumulation Steps:  {accumulation_steps}")
        print(f"Effective Batch:     {train_loader.batch_size * accumulation_steps}")
        print(f"Train samples:       {self.train_size}")
        print(f"Val samples:         {self.val_size}")
        print(f"Test samples:        {self.test_size}")
        print(f"{'='*60}\n")
    
    def train_one_epoch(self, epoch: int) -> Tuple[float, float]:
        """
        Train for one epoch.
        
        Args:
            epoch: Current epoch number
            
        Returns:
            Tuple of (average loss, accuracy)
        """
        self.model.train()
        
        running_loss = 0.0
        correct = 0
        total = 0
        
        # Progress bar
        pbar = tqdm(
            self.train_loader,
            desc=f"Epoch {epoch+1} [Train]",
            ncols=100,
            bar_format='{l_bar}{bar:20}{r_bar}{bar:-20b}'
        )
        
        self.optimizer.zero_grad()
        
        for batch_idx, (images, labels) in enumerate(pbar):
            images, labels = images.to(self.device), labels.to(self.device)
            
            # Mixed precision forward pass
            if self.use_amp:
                with autocast():
                    outputs = self.model(images)
                    loss = self.criterion(outputs, labels)
                    loss = loss / self.accumulation_steps
                
                # Backward with gradient scaling
                self.scaler.scale(loss).backward()
                
                # Optimizer step with accumulation
                if (batch_idx + 1) % self.accumulation_steps == 0:
                    # Gradient clipping (must unscale first with AMP)
                    if self.clip_grad_norm is not None:
                        self.scaler.unscale_(self.optimizer)
                        torch.nn.utils.clip_grad_norm_(
                            self.model.parameters(), 
                            max_norm=self.clip_grad_norm
                        )
                    
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                    self.optimizer.zero_grad()
            else:
                # Standard training
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                loss = loss / self.accumulation_steps
                
                loss.backward()
                
                if (batch_idx + 1) % self.accumulation_steps == 0:
                    # Gradient clipping (standard, no unscaling needed)
                    if self.clip_grad_norm is not None:
                        torch.nn.utils.clip_grad_norm_(
                            self.model.parameters(), 
                            max_norm=self.clip_grad_norm
                        )
                    
                    self.optimizer.step()
                    self.optimizer.zero_grad()
            
            # Metrics
            running_loss += loss.item() * self.accumulation_steps * images.size(0)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            # Update progress bar
            current_acc = 100.0 * correct / total
            pbar.set_postfix({
                'loss': f'{running_loss/total:.4f}',
                'acc': f'{current_acc:.2f}%'
            })
        
        epoch_loss = running_loss / total
        epoch_acc = 100.0 * correct / total
        
        return epoch_loss, epoch_acc
    
    def validate(self, epoch: int) -> Tuple[float, float]:
        """
        Validate the model.
        
        Args:
            epoch: Current epoch number
            
        Returns:
            Tuple of (average loss, accuracy)
        """
        self.model.eval()
        
        running_loss = 0.0
        correct = 0
        total = 0
        
        # Progress bar
        pbar = tqdm(
            self.val_loader,
            desc=f"Epoch {epoch+1} [Val]  ",
            ncols=100,
            bar_format='{l_bar}{bar:20}{r_bar}{bar:-20b}'
        )
        
        with torch.no_grad():
            for images, labels in pbar:
                images, labels = images.to(self.device), labels.to(self.device)
                
                # Mixed precision inference
                if self.use_amp:
                    with autocast():
                        outputs = self.model(images)
                        loss = self.criterion(outputs, labels)
                else:
                    outputs = self.model(images)
                    loss = self.criterion(outputs, labels)
                
                # Metrics
                running_loss += loss.item() * images.size(0)
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
                
                # Update progress bar
                current_acc = 100.0 * correct / total
                pbar.set_postfix({
                    'loss': f'{running_loss/total:.4f}',
                    'acc': f'{current_acc:.2f}%'
                })
        
        epoch_loss = running_loss / total
        epoch_acc = 100.0 * correct / total
        
        return epoch_loss, epoch_acc
    
    def save_checkpoint(
        self,
        epoch: int,
        val_loss: float,
        is_best: bool = False,
        filename: Optional[str] = None
    ):
        """
        Save model checkpoint with full training state.
        
        Args:
            epoch: Current epoch
            val_loss: Current validation loss
            is_best: Whether this is the best model so far
            filename: Optional custom filename
        """
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_val_loss': self.best_val_loss,
            'best_val_acc': self.best_val_acc,
            'current_val_loss': val_loss,
            'epochs_without_improvement': self.epochs_without_improvement,
            'history': self.history,
            'accumulation_steps': self.accumulation_steps,
            'timestamp': datetime.now().isoformat()
        }
        
        if self.scaler is not None:
            checkpoint['scaler_state_dict'] = self.scaler.state_dict()
        
        # Generate timestamped filename if not provided
        if filename is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f'{self.model_name}_epoch{epoch}_{timestamp}.pth'
        
        checkpoint_path = self.checkpoint_dir / filename
        
        try:
            torch.save(checkpoint, checkpoint_path)
            print(f"💾 Checkpoint saved: {checkpoint_path}")
            
            # Save/update best model with symlink behavior
            if is_best:
                best_path = self.checkpoint_dir / 'best.pth'
                
                # Save best model
                torch.save(checkpoint, best_path)
                
                # Also create a named best model copy
                best_named_path = self.checkpoint_dir / f'{self.model_name}_best.pth'
                shutil.copy2(best_path, best_named_path)
                
                print(f"✅ New best model saved: {best_path}")
                print(f"   Val Loss: {val_loss:.4f} (improved by {self.best_val_loss - val_loss:.4f})")
                
        except Exception as e:
            print(f"❌ Failed to save checkpoint: {e}")
            print(f"   Attempting to save to backup location...")
            
            # Try backup location
            backup_path = Path('.') / 'checkpoint_backup.pth'
            try:
                torch.save(checkpoint, backup_path)
                print(f"✅ Checkpoint saved to backup: {backup_path}")
            except Exception as e2:
                print(f"❌ Backup save also failed: {e2}")
    
    def load_checkpoint(self, checkpoint_path: str) -> bool:
        """
        Load checkpoint to resume training with error handling.
        
        Args:
            checkpoint_path: Path to checkpoint file
            
        Returns:
            bool: True if successful, False otherwise
        """
        checkpoint_path = Path(checkpoint_path)
        
        if not checkpoint_path.exists():
            print(f"❌ Checkpoint not found: {checkpoint_path}")
            return False
        
        try:
            print(f"📂 Loading checkpoint: {checkpoint_path}")
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            
            # Validate checkpoint
            required_keys = ['model_state_dict', 'optimizer_state_dict', 'epoch']
            missing_keys = [key for key in required_keys if key not in checkpoint]
            
            if missing_keys:
                print(f"❌ Corrupted checkpoint (missing keys: {missing_keys})")
                return False
            
            # Load model state
            self.model.load_state_dict(checkpoint['model_state_dict'])
            print("✅ Model state loaded")
            
            # Load optimizer state
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            print("✅ Optimizer state loaded")
            
            # Load scheduler state
            if 'scheduler_state_dict' in checkpoint:
                self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
                print("✅ Scheduler state loaded")
            
            # Load scaler state (AMP)
            if self.scaler is not None and 'scaler_state_dict' in checkpoint:
                self.scaler.load_state_dict(checkpoint['scaler_state_dict'])
                print("✅ AMP scaler state loaded")
            
            # Load training state
            self.current_epoch = checkpoint['epoch']
            self.best_val_loss = checkpoint.get('best_val_loss', float('inf'))
            self.best_val_acc = checkpoint.get('best_val_acc', 0.0)
            self.epochs_without_improvement = checkpoint.get('epochs_without_improvement', 0)
            self.history = checkpoint.get('history', {
                'train_loss': [], 'train_acc': [],
                'val_loss': [], 'val_acc': [],
                'learning_rate': []
            })
            
            print(f"\n{'='*60}")
            print("✅ CHECKPOINT LOADED SUCCESSFULLY")
            print(f"{'='*60}")
            print(f"Resuming from epoch:     {self.current_epoch}")
            print(f"Best val loss:           {self.best_val_loss:.4f}")
            print(f"Best val acc:            {self.best_val_acc:.2f}%")
            print(f"Epochs without improve:  {self.epochs_without_improvement}")
            
            if 'timestamp' in checkpoint:
                print(f"Checkpoint timestamp:    {checkpoint['timestamp']}")
            
            print(f"{'='*60}\n")
            
            return True
            
        except Exception as e:
            print(f"❌ Failed to load checkpoint: {e}")
            print(f"   Error type: {type(e).__name__}")
            print(f"   Checkpoint may be corrupted or incompatible")
            return False
    
    def save_history(self):
        """Save training history to JSON."""
        history_path = self.log_dir / 'training_history.json'
        
        with open(history_path, 'w') as f:
            json.dump(self.history, f, indent=4)
        
        print(f"📊 Training history saved: {history_path}")
    
    def train(
        self,
        max_epochs: int,
        early_stopping_patience: Optional[int] = None,
        save_every: int = 10
    ):
        """
        Main training loop.
        
        Args:
            max_epochs: Maximum number of epochs
            early_stopping_patience: Stop if no improvement for N epochs
            save_every: Save checkpoint every N epochs
        """
        print(f"\n{'='*60}")
        print(f"🎯 TRAINING STARTED")
        print(f"{'='*60}")
        print(f"Max epochs:          {max_epochs}")
        print(f"Early stopping:      {early_stopping_patience if early_stopping_patience else 'Disabled'}")
        print(f"Save every:          {save_every} epochs")
        print(f"{'='*60}\n")
        
        start_time = time.time()
        epochs_without_improvement = 0
        
        for epoch in range(self.current_epoch, max_epochs):
            epoch_start = time.time()
            
            # Train
            train_loss, train_acc = self.train_one_epoch(epoch)
            
            # Validate
            val_loss, val_acc = self.validate(epoch)
            
            # Update learning rate (correct step() placement based on scheduler type)
            if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                # ReduceLROnPlateau needs validation loss
                self.scheduler.step(val_loss)
            elif hasattr(self.scheduler, 'step'):
                # For StepLR, CosineAnnealingLR, and warmup wrapper
                # Check if warmup wrapper (has step() that accepts metrics)
                if hasattr(self.scheduler, 'scheduler') and isinstance(
                    self.scheduler.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau
                ):
                    # Warmup wrapper with ReduceLROnPlateau inside
                    self.scheduler.step(val_loss)
                else:
                    # Standard schedulers (StepLR, CosineAnnealingLR, warmup)
                    self.scheduler.step()
            
            current_lr = self.optimizer.param_groups[0]['lr']
            
            # Log metrics
            self.history['train_loss'].append(train_loss)
            self.history['train_acc'].append(train_acc)
            self.history['val_loss'].append(val_loss)
            self.history['val_acc'].append(val_acc)
            self.history['learning_rate'].append(current_lr)
            
            # Print epoch summary
            epoch_time = time.time() - epoch_start
            print(f"\nEpoch {epoch+1}/{max_epochs} Summary:")
            print(f"  Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
            print(f"  Val Loss:   {val_loss:.4f} | Val Acc:   {val_acc:.2f}%")
            print(f"  LR: {current_lr:.6f} | Time: {epoch_time:.1f}s")
            
            # Check for best model (based on val loss)
            is_best = val_loss < self.best_val_loss
            if is_best:
                self.best_val_loss = val_loss
                self.best_val_acc = val_acc
                self.epochs_without_improvement = 0
                print(f"  🏆 New best model! Val Loss: {val_loss:.4f}")
            else:
                self.epochs_without_improvement += 1
            
            # Save checkpoint (pass val_loss for enhanced tracking)
            if (epoch + 1) % save_every == 0 or is_best:
                self.save_checkpoint(epoch + 1, val_loss=val_loss, is_best=is_best)
            
            # Early stopping
            if early_stopping_patience and self.epochs_without_improvement >= early_stopping_patience:
                print(f"\n⚠️  Early stopping triggered (no improvement for {early_stopping_patience} epochs)")
                break
            
            print(f"{'-'*60}")
        
        # Training complete
        total_time = time.time() - start_time
        print(f"\n{'='*60}")
        print(f"✅ TRAINING COMPLETE")
        print(f"{'='*60}")
        print(f"Total time:          {total_time/60:.1f} minutes")
        print(f"Best val loss:       {self.best_val_loss:.4f}")
        print(f"Best val acc:        {self.best_val_acc:.2f}%")
        print(f"{'='*60}\n")
        
        # Save final checkpoint and history
        self.save_checkpoint(epoch + 1, val_loss=val_loss, filename=f'{self.model_name}_final.pth')
        self.save_history()


def freeze_backbone(model: nn.Module, model_type: str):
    """
    Freeze backbone layers for transfer learning.
    
    Args:
        model: PyTorch model
        model_type: Model type ('custom' or 'efficientnet')
    """
    if model_type.lower() in ['efficientnet', 'efficientnet_b0']:
        # Freeze all layers except classifier
        for name, param in model.named_parameters():
            if 'classifier' not in name:
                param.requires_grad = False
        
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in model.parameters())
        print(f"🔒 Backbone frozen: {trainable:,}/{total:,} parameters trainable")
    else:
        print("⚠️  Backbone freezing only supported for transfer learning models")


def resume_from_checkpoint(
    checkpoint_path: str,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler._LRScheduler,
    scaler: Optional[torch.cuda.amp.GradScaler] = None,
    device: torch.device = None
) -> Tuple[nn.Module, torch.optim.Optimizer, torch.optim.lr_scheduler._LRScheduler, 
           Optional[torch.cuda.amp.GradScaler], int, float, float, int, dict]:
    """
    Resume training from a checkpoint with graceful error handling.
    
    Args:
        checkpoint_path: Path to checkpoint file
        model: Model to load state into
        optimizer: Optimizer to load state into
        scheduler: Learning rate scheduler to load state into
        scaler: Optional AMP GradScaler to load state into
        device: Device to map checkpoint to
        
    Returns:
        Tuple containing:
        - model: Model with loaded state
        - optimizer: Optimizer with loaded state
        - scheduler: Scheduler with loaded state
        - scaler: GradScaler with loaded state (if provided)
        - start_epoch: Epoch to resume from
        - best_val_loss: Best validation loss from checkpoint
        - best_val_acc: Best validation accuracy from checkpoint
        - epochs_without_improvement: Epochs without improvement counter
        - history: Training history dict
        
    Example:
        >>> # Setup model and components
        >>> model, optimizer, criterion, scheduler, device = get_model('custom')
        >>> scaler = torch.cuda.amp.GradScaler() if device.type == 'cuda' else None
        >>> 
        >>> # Resume from checkpoint
        >>> checkpoint_path = 'checkpoints/best.pth'
        >>> model, optimizer, scheduler, scaler, start_epoch, best_val_loss, \
        ...     best_val_acc, epochs_without_improvement, history = resume_from_checkpoint(
        ...     checkpoint_path, model, optimizer, scheduler, scaler, device
        ... )
        >>> 
        >>> # Create trainer and continue training
        >>> trainer = Trainer(model, optimizer, criterion, scheduler, device, 
        ...                   train_loader, val_loader, test_loader, scaler=scaler)
        >>> trainer.current_epoch = start_epoch
        >>> trainer.best_val_loss = best_val_loss
        >>> trainer.best_val_acc = best_val_acc
        >>> trainer.epochs_without_improvement = epochs_without_improvement
        >>> trainer.history = history
        >>> trainer.train(max_epochs=100, early_stopping_patience=5)
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    checkpoint_path = Path(checkpoint_path)
    
    # Check if file exists
    if not checkpoint_path.exists():
        print(f"❌ Checkpoint file not found: {checkpoint_path}")
        print(f"   Starting training from scratch...")
        return model, optimizer, scheduler, scaler, 0, float('inf'), 0.0, 0, {
            'train_loss': [], 'train_acc': [],
            'val_loss': [], 'val_acc': [],
            'learning_rate': []
        }
    
    try:
        print(f"📂 Loading checkpoint: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=device)
        
        # Validate checkpoint integrity
        required_keys = ['model_state_dict', 'optimizer_state_dict', 'epoch']
        missing_keys = [key for key in required_keys if key not in checkpoint]
        
        if missing_keys:
            print(f"❌ Corrupted checkpoint (missing required keys: {missing_keys})")
            print(f"   Starting training from scratch...")
            return model, optimizer, scheduler, scaler, 0, float('inf'), 0.0, 0, {
                'train_loss': [], 'train_acc': [],
                'val_loss': [], 'val_acc': [],
                'learning_rate': []
            }
        
        # Load model state
        model.load_state_dict(checkpoint['model_state_dict'])
        print("✅ Model state loaded")
        
        # Load optimizer state
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        print("✅ Optimizer state loaded")
        
        # Load scheduler state
        if 'scheduler_state_dict' in checkpoint:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            print("✅ Scheduler state loaded")
        
        # Load scaler state (AMP)
        if scaler is not None and 'scaler_state_dict' in checkpoint:
            scaler.load_state_dict(checkpoint['scaler_state_dict'])
            print("✅ AMP scaler state loaded")
        
        # Extract training state
        start_epoch = checkpoint['epoch']
        best_val_loss = checkpoint.get('best_val_loss', float('inf'))
        best_val_acc = checkpoint.get('best_val_acc', 0.0)
        epochs_without_improvement = checkpoint.get('epochs_without_improvement', 0)
        history = checkpoint.get('history', {
            'train_loss': [], 'train_acc': [],
            'val_loss': [], 'val_acc': [],
            'learning_rate': []
        })
        
        print(f"\n{'='*60}")
        print("✅ CHECKPOINT LOADED SUCCESSFULLY")
        print(f"{'='*60}")
        print(f"Resuming from epoch:     {start_epoch}")
        print(f"Best val loss:           {best_val_loss:.4f}")
        print(f"Best val acc:            {best_val_acc:.2f}%")
        print(f"Epochs w/o improvement:  {epochs_without_improvement}")
        
        if 'timestamp' in checkpoint:
            print(f"Checkpoint timestamp:    {checkpoint['timestamp']}")
        
        print(f"{'='*60}\n")
        
        return model, optimizer, scheduler, scaler, start_epoch, best_val_loss, \
               best_val_acc, epochs_without_improvement, history
        
    except Exception as e:
        print(f"❌ Failed to load checkpoint: {e}")
        print(f"   Error type: {type(e).__name__}")
        print(f"   Checkpoint may be corrupted or incompatible")
        print(f"   Starting training from scratch...")
        return model, optimizer, scheduler, scaler, 0, float('inf'), 0.0, 0, {
            'train_loss': [], 'train_acc': [],
            'val_loss': [], 'val_acc': [],
            'learning_rate': []
        }


def main(args):
    """Main training function."""
    print(f"\n{'='*60}")
    print("🌾 CROPSHIELD AI - TRAINING")
    print(f"{'='*60}")
    print(f"Model:               {args.model}")
    print(f"Epochs:              {args.epochs}")
    print(f"Batch size:          {args.batch_size}")
    print(f"Accumulation steps:  {args.accumulation_steps}")
    print(f"Effective batch:     {args.batch_size * args.accumulation_steps}")
    print(f"Learning rate:       {args.lr}")
    print(f"Freeze backbone:     {args.freeze_backbone}")
    print(f"Mixed precision:     {args.amp}")
    print(f"{'='*60}\n")
    
    # 1. Load data
    print("📦 Loading dataset...")
    train_loader, val_loader, test_loader, class_names, info = make_loaders(
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        augmentation_mode=args.augmentation_mode
    )
    
    num_classes = len(class_names)
    print(f"✅ Dataset loaded: {num_classes} classes, {info['total_images']:,} images")
    
    # 2. Create model
    print(f"\n📦 Creating model: {args.model}")
    model, optimizer, criterion, scheduler, device = get_model(
        model_type=args.model,
        num_classes=num_classes,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        scheduler_type=args.scheduler,
        scheduler_step_size=args.scheduler_step_size,
        scheduler_gamma=args.scheduler_gamma,
        scheduler_patience=args.scheduler_patience,
        scheduler_t_max=args.scheduler_t_max,
        warmup_epochs=args.warmup_epochs,
        verbose=True
    )
    
    # 3. Freeze backbone if requested
    if args.freeze_backbone:
        print("\n🔒 Freezing backbone layers...")
        freeze_backbone(model, args.model)
    
    # 4. Create trainer
    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        criterion=criterion,
        scheduler=scheduler,
        device=device,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        num_classes=num_classes,
        accumulation_steps=args.accumulation_steps,
        use_amp=args.amp,
        clip_grad_norm=args.clip_grad_norm,
        checkpoint_dir=args.checkpoint_dir,
        log_dir=args.log_dir,
        model_name=f"{args.model}_{args.run_name}" if args.run_name else args.model
    )
    
    # 5. Resume from checkpoint if provided
    if args.resume:
        print(f"\n📂 Loading checkpoint: {args.resume}")
        trainer.load_checkpoint(args.resume)
    
    # 6. Train
    trainer.train(
        max_epochs=args.epochs,
        early_stopping_patience=args.early_stopping,
        save_every=args.save_every
    )
    
    # 7. Final evaluation on test set
    print("\n📊 Evaluating on test set...")
    test_loss, test_acc = trainer.validate(epoch=args.epochs)
    print(f"Test Loss: {test_loss:.4f} | Test Acc: {test_acc:.2f}%")
    
    print("\n✅ Training pipeline complete!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train CropShield AI plant disease classification model"
    )
    
    # Model arguments
    parser.add_argument(
        '--model',
        type=str,
        default='custom',
        choices=['custom', 'efficientnet', 'efficientnet_b0'],
        help='Model architecture (default: custom)'
    )
    parser.add_argument(
        '--freeze_backbone',
        action='store_true',
        help='Freeze backbone layers (transfer learning only)'
    )
    
    # Training arguments
    parser.add_argument(
        '--epochs',
        type=int,
        default=25,
        help='Maximum number of epochs (default: 25)'
    )
    parser.add_argument(
        '--batch_size',
        type=int,
        default=32,
        help='Batch size per GPU (default: 32)'
    )
    parser.add_argument(
        '--accumulation_steps',
        type=int,
        default=1,
        help='Gradient accumulation steps (default: 1)'
    )
    parser.add_argument(
        '--lr',
        type=float,
        default=1e-4,
        help='Learning rate (default: 1e-4)'
    )
    parser.add_argument(
        '--weight_decay',
        type=float,
        default=1e-4,
        help='Weight decay (default: 1e-4)'
    )
    
    # Learning rate scheduler arguments
    parser.add_argument(
        '--scheduler',
        type=str,
        default='step',
        choices=['step', 'plateau', 'cosine'],
        help='LR scheduler type: step (StepLR), plateau (ReduceLROnPlateau), cosine (CosineAnnealingLR) (default: step)'
    )
    parser.add_argument(
        '--warmup_epochs',
        type=int,
        default=0,
        help='Linear warmup epochs (0 to disable). Recommended: 0 for custom, 1 for pretrained (default: 0)'
    )
    parser.add_argument(
        '--scheduler_step_size',
        type=int,
        default=5,
        help='StepLR: Epochs between LR reduction (default: 5)'
    )
    parser.add_argument(
        '--scheduler_gamma',
        type=float,
        default=0.5,
        help='StepLR/Plateau: LR reduction factor (default: 0.5)'
    )
    parser.add_argument(
        '--scheduler_patience',
        type=int,
        default=3,
        help='ReduceLROnPlateau: Patience epochs before reducing LR (default: 3)'
    )
    parser.add_argument(
        '--scheduler_t_max',
        type=int,
        default=50,
        help='CosineAnnealingLR: Maximum epochs for one cycle (default: 50)'
    )
    
    # Data arguments
    parser.add_argument(
        '--num_workers',
        type=int,
        default=8,
        help='Number of data loading workers (default: 8)'
    )
    parser.add_argument(
        '--augmentation_mode',
        type=str,
        default='moderate',
        choices=['light', 'moderate', 'aggressive'],
        help='Data augmentation mode (default: moderate)'
    )
    
    # Training optimization
    parser.add_argument(
        '--amp',
        type=bool,
        default=True,
        help='Use mixed precision training (default: True)'
    )
    parser.add_argument(
        '--clip_grad_norm',
        type=float,
        default=None,
        help='Gradient clipping max norm (None to disable). Recommended: 1.0 for unstable training (default: None)'
    )
    parser.add_argument(
        '--early_stopping',
        type=int,
        default=None,
        help='Early stopping patience (epochs). None to disable (default: None)'
    )
    
    # Checkpointing
    parser.add_argument(
        '--checkpoint_dir',
        type=str,
        default='models',
        help='Directory to save checkpoints (default: models)'
    )
    parser.add_argument(
        '--log_dir',
        type=str,
        default='.',
        help='Directory to save logs (default: .)'
    )
    parser.add_argument(
        '--save_every',
        type=int,
        default=10,
        help='Save checkpoint every N epochs (default: 10)'
    )
    parser.add_argument(
        '--resume',
        type=str,
        default=None,
        help='Path to checkpoint to resume from'
    )
    parser.add_argument(
        '--run_name',
        type=str,
        default=None,
        help='Run name for logging (default: None)'
    )
    
    args = parser.parse_args()
    
    # Run training
    main(args)
