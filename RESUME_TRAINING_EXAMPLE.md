# Resume Training Example

This guide demonstrates how to resume training from a checkpoint in CropShield AI.

## Overview

The training script now includes:
- ✅ **Early stopping** with configurable patience (default: 5 epochs)
- ✅ **Timestamped checkpoints** for history preservation
- ✅ **best.pth** symlink-style behavior for easy access to best model
- ✅ **Graceful error handling** for missing/corrupted checkpoints
- ✅ **Complete state preservation** (model, optimizer, scheduler, scaler, epoch, history)

## Checkpoint Structure

Checkpoints are saved in the `checkpoints/` directory:

```
checkpoints/
├── best.pth                              # Latest best model (easy access)
├── custom_best.pth                       # Named copy of best model
├── custom_epoch10_20250109_143022.pth    # Timestamped checkpoint
├── custom_epoch20_20250109_144530.pth    # Another timestamped checkpoint
└── custom_final.pth                      # Final model after training
```

Each checkpoint contains:
- `model_state_dict`: Model weights
- `optimizer_state_dict`: Optimizer state (momentum, etc.)
- `scheduler_state_dict`: Learning rate scheduler state
- `scaler_state_dict`: AMP GradScaler state (if using mixed precision)
- `epoch`: Current epoch number
- `best_val_loss`: Best validation loss so far
- `best_val_acc`: Best validation accuracy
- `current_val_loss`: Validation loss at this checkpoint
- `epochs_without_improvement`: Counter for early stopping
- `history`: Training history (losses, accuracies, learning rates)
- `timestamp`: Checkpoint creation timestamp

## Method 1: Resume via CLI (Recommended)

The simplest way to resume training:

```bash
# Start initial training
python train.py --model custom --epochs 100 --early_stopping 5 --batch_size 32

# If training stops (interrupted, early stopped, etc.), resume from best checkpoint
python train.py --model custom --epochs 100 --early_stopping 5 --batch_size 32 --resume checkpoints/best.pth

# Or resume from a specific timestamped checkpoint
python train.py --model custom --epochs 100 --early_stopping 5 --batch_size 32 --resume checkpoints/custom_epoch20_20250109_144530.pth
```

### Key Points:
- Training resumes from the saved epoch
- All optimizer/scheduler states are preserved
- Training history is restored (for seamless plotting)
- If checkpoint is missing/corrupted, training starts from scratch with a warning

## Method 2: Resume Programmatically

Use the `resume_from_checkpoint()` utility function:

```python
from train import Trainer, resume_from_checkpoint
from models.model_factory import get_model
from fast_dataset import make_loaders
import torch

# 1. Setup data loaders
train_loader, val_loader, test_loader, class_names, info = make_loaders(
    batch_size=32,
    num_workers=8,
    augmentation_mode='moderate'
)

num_classes = len(class_names)

# 2. Create model and components
model, optimizer, criterion, scheduler, device = get_model(
    model_type='custom',
    num_classes=num_classes,
    learning_rate=1e-4,
    weight_decay=1e-4
)

# 3. Setup AMP scaler (if using mixed precision)
scaler = torch.cuda.amp.GradScaler() if device.type == 'cuda' else None

# 4. Resume from checkpoint
checkpoint_path = 'checkpoints/best.pth'
model, optimizer, scheduler, scaler, start_epoch, best_val_loss, \
    best_val_acc, epochs_without_improvement, history = resume_from_checkpoint(
    checkpoint_path=checkpoint_path,
    model=model,
    optimizer=optimizer,
    scheduler=scheduler,
    scaler=scaler,
    device=device
)

# 5. Create trainer and restore state
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
    use_amp=True,
    checkpoint_dir='checkpoints',
    model_name='custom'
)

# Restore training state
trainer.current_epoch = start_epoch
trainer.best_val_loss = best_val_loss
trainer.best_val_acc = best_val_acc
trainer.epochs_without_improvement = epochs_without_improvement
trainer.history = history

# 6. Continue training
trainer.train(max_epochs=100, early_stopping_patience=5)
```

## Method 3: Using Trainer's load_checkpoint()

For finer control within the Trainer class:

```python
from train import Trainer
from models.model_factory import get_model
from fast_dataset import make_loaders

# Setup
train_loader, val_loader, test_loader, class_names, info = make_loaders(batch_size=32)
model, optimizer, criterion, scheduler, device = get_model('custom', len(class_names))

# Create trainer
trainer = Trainer(
    model=model,
    optimizer=optimizer,
    criterion=criterion,
    scheduler=scheduler,
    device=device,
    train_loader=train_loader,
    val_loader=val_loader,
    test_loader=test_loader,
    num_classes=len(class_names),
    use_amp=True
)

# Load checkpoint (returns True if successful, False otherwise)
success = trainer.load_checkpoint('checkpoints/best.pth')

if success:
    print("✅ Checkpoint loaded, continuing training...")
    trainer.train(max_epochs=100, early_stopping_patience=5)
else:
    print("⚠️  No checkpoint loaded, starting from scratch...")
    trainer.train(max_epochs=100, early_stopping_patience=5)
```

## Error Handling Examples

### Missing Checkpoint
```python
# If checkpoint doesn't exist
checkpoint_path = 'checkpoints/nonexistent.pth'
model, optimizer, scheduler, scaler, start_epoch, best_val_loss, \
    best_val_acc, epochs_without_improvement, history = resume_from_checkpoint(
    checkpoint_path, model, optimizer, scheduler, scaler, device
)
# Output: ❌ Checkpoint file not found: checkpoints/nonexistent.pth
#         Starting training from scratch...
# Returns: start_epoch=0, best_val_loss=inf, empty history
```

### Corrupted Checkpoint
```python
# If checkpoint is corrupted (missing required keys)
checkpoint_path = 'checkpoints/corrupted.pth'
model, optimizer, scheduler, scaler, start_epoch, best_val_loss, \
    best_val_acc, epochs_without_improvement, history = resume_from_checkpoint(
    checkpoint_path, model, optimizer, scheduler, scaler, device
)
# Output: ❌ Corrupted checkpoint (missing required keys: ['model_state_dict'])
#         Starting training from scratch...
# Returns: start_epoch=0, best_val_loss=inf, empty history
```

### General Error
```python
# If any other error occurs during loading
checkpoint_path = 'checkpoints/incompatible.pth'
model, optimizer, scheduler, scaler, start_epoch, best_val_loss, \
    best_val_acc, epochs_without_improvement, history = resume_from_checkpoint(
    checkpoint_path, model, optimizer, scheduler, scaler, device
)
# Output: ❌ Failed to load checkpoint: <error details>
#         Error type: RuntimeError
#         Checkpoint may be corrupted or incompatible
#         Starting training from scratch...
# Returns: start_epoch=0, best_val_loss=inf, empty history
```

## Early Stopping Example

Train with early stopping (patience=5):

```bash
# Training will automatically stop if no improvement for 5 consecutive epochs
python train.py --model custom --epochs 100 --early_stopping 5 --batch_size 32
```

Example output:
```
Epoch 45/100 Summary:
  Train Loss: 0.3421 | Train Acc: 88.45%
  Val Loss:   0.4521 | Val Acc:   85.23%
  LR: 0.000100 | Time: 95.3s
  Epochs w/o improvement: 5/5
------------------------------------------------------------

============================================================
⛔ EARLY STOPPING TRIGGERED
============================================================
No improvement in validation loss for 5 epochs
Best val loss: 0.4123 (epoch 40)
Best val acc:  86.78%
Stopping at epoch 45/100
============================================================
```

## Common Use Cases

### 1. Long Training Run (Resumable)
```bash
# Start training for 100 epochs with early stopping
python train.py --model custom --epochs 100 --early_stopping 5 --batch_size 32

# If interrupted (power loss, Ctrl+C, etc.), resume from best checkpoint
python train.py --model custom --epochs 100 --early_stopping 5 --batch_size 32 --resume checkpoints/best.pth
```

### 2. Extend Training After Early Stop
```bash
# Initial training with early stopping
python train.py --model custom --epochs 50 --early_stopping 5 --batch_size 32

# Early stopped at epoch 30. Want to continue with different learning rate?
# Load checkpoint, adjust LR, continue training
python train.py --model custom --epochs 80 --early_stopping 5 --batch_size 32 --lr 1e-5 --resume checkpoints/best.pth
```

### 3. Fine-tune from Best Model
```bash
# Train base model
python train.py --model custom --epochs 50 --batch_size 32

# Fine-tune from best checkpoint with lower LR
python train.py --model custom --epochs 100 --batch_size 32 --lr 1e-5 --resume checkpoints/best.pth
```

### 4. Experiment with Different Hyperparameters
```bash
# Baseline training
python train.py --model custom --epochs 50 --batch_size 32 --augmentation_mode moderate

# Load best model, try aggressive augmentation
python train.py --model custom --epochs 100 --batch_size 32 --augmentation_mode aggressive --resume checkpoints/best.pth
```

## Best Practices

1. **Always use `--early_stopping`** for long training runs to prevent overfitting
   ```bash
   python train.py --model custom --epochs 100 --early_stopping 5
   ```

2. **Use descriptive run names** for organized checkpoints
   ```bash
   python train.py --model custom --epochs 100 --run_name "baseline_lr1e4"
   # Creates: custom_baseline_lr1e4_epoch10_*.pth
   ```

3. **Save frequently on long runs** with `--save_every`
   ```bash
   python train.py --model custom --epochs 100 --save_every 5
   # Saves every 5 epochs + whenever best model improves
   ```

4. **Monitor checkpoint directory size** - timestamped files accumulate over time
   ```bash
   # Keep only best and latest 3 checkpoints, remove old ones
   ls -t checkpoints/*.pth | tail -n +4 | grep -v best.pth | xargs rm
   ```

5. **Always resume from `best.pth`** for consistent best-model access
   ```bash
   python train.py --model custom --epochs 100 --resume checkpoints/best.pth
   ```

6. **Use programmatic resume** for custom workflows (hyperparameter tuning, ensemble training)

## Troubleshooting

**Q: Can I resume training with different hyperparameters?**  
A: Yes! The model weights, optimizer momentum, and learning rate schedule are restored. You can then modify learning rate, batch size, augmentation, etc.

**Q: What happens if I resume with a different model architecture?**  
A: The checkpoint loading will fail with a state dict mismatch error. Training will start from scratch with a warning.

**Q: Can I resume training on a different GPU?**  
A: Yes! Checkpoints are saved with `map_location=device` parameter, allowing transfer between CPU/GPU and different GPUs.

**Q: Will training history plots be continuous after resume?**  
A: Yes! The `history` dict is restored, so plotting will show seamless training curves across resume points.

**Q: How do I know if a checkpoint was loaded successfully?**  
A: Check the console output. Successful load shows:
```
✅ CHECKPOINT LOADED SUCCESSFULLY
============================================================
Resuming from epoch:     25
Best val loss:           0.4123
Best val acc:            86.78%
...
```

Failed load shows:
```
❌ Checkpoint not found: checkpoints/missing.pth
   Starting training from scratch...
```

## Summary

The CropShield AI training script provides robust checkpoint management:

✅ **Automatic checkpointing** with timestamped files  
✅ **Early stopping** to prevent overfitting  
✅ **Graceful resume** from interruptions/errors  
✅ **Complete state preservation** for perfect resume  
✅ **Easy access** via `best.pth` symlink  
✅ **Error handling** for missing/corrupted files  
✅ **CLI and programmatic** interfaces  

Use `--resume checkpoints/best.pth` to resume from the best model, or use the `resume_from_checkpoint()` utility for custom workflows!
