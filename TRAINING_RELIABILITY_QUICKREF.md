# Training Reliability - Quick Reference

## 🚀 Quick Start Commands

### Start Training with Reliability Features
```bash
# Custom CNN with early stopping (patience=5)
python train.py --model custom --epochs 25 --early_stopping 5 --batch_size 32

# EfficientNet-B0 with early stopping
python train.py --model efficientnet_b0 --epochs 25 --early_stopping 5 --batch_size 32
```

### Resume from Checkpoint
```bash
# Resume from best checkpoint
python train.py --model custom --epochs 100 --resume checkpoints/best.pth

# Resume from specific checkpoint
python train.py --model custom --epochs 100 --resume checkpoints/custom_epoch20_20250109_144530.pth
```

## 📂 Checkpoint Files

```
checkpoints/
├── best.pth                              # Latest best model (use this for resume)
├── custom_best.pth                       # Named copy of best model
├── custom_epoch10_20250109_143022.pth    # Timestamped checkpoint
└── custom_final.pth                      # Final model after training
```

## ⚡ Common Workflows

### Long Training with Auto-Save
```bash
# Save every 5 epochs + on improvement
python train.py --model custom --epochs 100 --early_stopping 5 --save_every 5 --batch_size 32
```

### Experiment with Named Run
```bash
# Descriptive run name for organization
python train.py --model custom --epochs 50 --run_name "baseline_lr1e4" --batch_size 32
# Creates: custom_baseline_lr1e4_epoch10_*.pth
```

### Fine-tune from Best Model
```bash
# Train baseline
python train.py --model custom --epochs 50 --batch_size 32

# Fine-tune with lower LR from best checkpoint
python train.py --model custom --epochs 100 --lr 1e-5 --resume checkpoints/best.pth
```

## 🛠️ Key Features

| Feature | Flag | Default | Description |
|---------|------|---------|-------------|
| **Early Stopping** | `--early_stopping N` | `None` | Stop if no improvement for N epochs |
| **Resume Training** | `--resume PATH` | `None` | Resume from checkpoint file |
| **Save Frequency** | `--save_every N` | `10` | Save checkpoint every N epochs |
| **Run Name** | `--run_name NAME` | `None` | Name for checkpoint organization |
| **Checkpoint Dir** | `--checkpoint_dir DIR` | `models` | Directory for checkpoints |

## 🧪 Test Commands

### Quick Test (2 epochs)
```bash
python train.py --model custom --epochs 2 --batch_size 32
```

### Test Early Stopping (patience=2)
```bash
python train.py --model custom --epochs 20 --early_stopping 2 --batch_size 32
```

### Test Resume
```bash
# Train 3 epochs
python train.py --model custom --epochs 3 --batch_size 32

# Resume to 6 epochs
python train.py --model custom --epochs 6 --resume checkpoints/best.pth
```

## 📊 Checkpoint Contents

Each checkpoint contains:
- ✅ Model weights (`model_state_dict`)
- ✅ Optimizer state (`optimizer_state_dict`)
- ✅ LR scheduler state (`scheduler_state_dict`)
- ✅ AMP scaler state (`scaler_state_dict`)
- ✅ Current epoch (`epoch`)
- ✅ Best validation loss/accuracy
- ✅ Training history (losses, accuracies, LRs)
- ✅ Early stopping counter
- ✅ Timestamp

## 🔧 Programmatic Resume

```python
from train import resume_from_checkpoint
from models.model_factory import get_model

# Setup
model, optimizer, criterion, scheduler, device = get_model('custom', 22)
scaler = torch.cuda.amp.GradScaler()

# Resume
model, optimizer, scheduler, scaler, start_epoch, best_val_loss, \
    best_val_acc, epochs_without_improvement, history = resume_from_checkpoint(
    'checkpoints/best.pth', model, optimizer, scheduler, scaler, device
)
```

## 🎯 Expected Behavior

### Early Stopping Triggered
```
Epoch 20/50 Summary:
  Train Loss: 0.3421 | Train Acc: 88.45%
  Val Loss:   0.4521 | Val Acc:   85.23%
  Epochs w/o improvement: 5/5

⛔ EARLY STOPPING TRIGGERED
No improvement in validation loss for 5 epochs
Best val loss: 0.4123 (epoch 15)
Stopping at epoch 20/50
```

### Checkpoint Saved
```
💾 Checkpoint saved: checkpoints/custom_epoch10_20250109_143022.pth
✅ New best model saved: checkpoints/best.pth
   Val Loss: 0.4123 (improved by 0.0234)
```

### Resume Success
```
✅ CHECKPOINT LOADED SUCCESSFULLY
Resuming from epoch:     20
Best val loss:           0.4123
Best val acc:            86.78%
```

### Resume Failure (Graceful)
```
❌ Checkpoint not found: checkpoints/missing.pth
   Starting training from scratch...
```

## 📚 Documentation

- **Comprehensive Guide**: `RESUME_TRAINING_EXAMPLE.md` (400+ lines)
- **Implementation Details**: `TRAINING_RELIABILITY_SUMMARY.md`
- **Main Training Guide**: `TRAINING_GUIDE.md`

## 🚨 Important Notes

1. Always use `--early_stopping` for long runs to prevent overfitting
2. Use `best.pth` for resume (always points to best checkpoint)
3. Checkpoints are saved when val loss improves + every N epochs
4. Resume works across interruptions (power loss, Ctrl+C, crashes)
5. Missing/corrupted checkpoints handled gracefully (warn + start from scratch)
6. All optimizer/scheduler states preserved for perfect resume

---

**For detailed examples and troubleshooting, see `RESUME_TRAINING_EXAMPLE.md`**
