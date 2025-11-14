# Training Script - Quick Reference

## 🚀 Quick Start

### Train Custom CNN (Default)
```bash
python train.py --model custom --epochs 25
```

### Train EfficientNet-B0 (Transfer Learning)
```bash
python train.py --model efficientnet --epochs 25
```

### Train with Frozen Backbone (Fine-tuning)
```bash
python train.py --model efficientnet --epochs 25 --freeze_backbone
```

---

## 📋 Common Training Commands

### 1. Quick Training (Custom CNN, 25 epochs)
```bash
python train.py --model custom --epochs 25 --batch_size 32
```
- **Expected time**: ~2-3 hours
- **Expected accuracy**: 75-85%

### 2. Transfer Learning (EfficientNet-B0)
```bash
python train.py --model efficientnet --epochs 25 --lr 1e-4
```
- **Expected time**: ~45 minutes
- **Expected accuracy**: 92-96%

### 3. With Gradient Accumulation (Simulate Larger Batch)
```bash
python train.py --model custom --epochs 25 --batch_size 16 --accumulation_steps 2
```
- **Effective batch size**: 16 × 2 = 32
- Use when GPU memory is limited

### 4. With Early Stopping
```bash
python train.py --model custom --epochs 100 --early_stopping 15
```
- Stops if no improvement for 15 epochs
- Prevents overfitting

### 5. Resume Training from Checkpoint
```bash
python train.py --model custom --epochs 50 --resume models/custom_checkpoint_epoch25.pth
```
- Continues from saved checkpoint
- Preserves optimizer state

### 6. Custom Run Name (for Multiple Experiments)
```bash
python train.py --model efficientnet --epochs 25 --run_name "exp1_frozen_backbone" --freeze_backbone
```
- Saves as: `efficientnet_exp1_frozen_backbone.pth`
- Useful for experiment tracking

---

## ⚙️ All Command-Line Arguments

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--model` | str | `custom` | Model: `custom`, `efficientnet`, `efficientnet_b0` |
| `--freeze_backbone` | flag | False | Freeze backbone (transfer learning only) |
| `--epochs` | int | 25 | Maximum training epochs |
| `--batch_size` | int | 32 | Batch size per GPU |
| `--accumulation_steps` | int | 1 | Gradient accumulation steps |
| `--lr` | float | 1e-4 | Learning rate |
| `--weight_decay` | float | 1e-4 | L2 regularization |
| `--num_workers` | int | 8 | Data loading workers |
| `--augmentation_mode` | str | `moderate` | Augmentation: `light`, `moderate`, `aggressive` |
| `--amp` | bool | True | Mixed precision training |
| `--early_stopping` | int | None | Early stopping patience (epochs) |
| `--checkpoint_dir` | str | `models` | Checkpoint save directory |
| `--log_dir` | str | `.` | Log save directory |
| `--save_every` | int | 10 | Save checkpoint every N epochs |
| `--resume` | str | None | Path to checkpoint for resuming |
| `--run_name` | str | None | Custom run name for logging |

---

## 🎯 Training Features

### ✅ Mixed Precision Training (AMP)
- **Enabled by default** on CUDA GPUs
- **2x faster training** on RTX 4060
- **Lower memory usage** (can use larger batches)
- Disable with `--amp False` if issues occur

### ✅ Gradient Accumulation
- Simulates larger batch sizes on limited GPU memory
- `--batch_size 16 --accumulation_steps 2` → effective batch = 32
- Useful when `batch_size=32` causes OOM

### ✅ Progress Bars (tqdm)
```
Epoch 1 [Train] ████████████████ 100%  loss: 2.1234 acc: 45.67%
Epoch 1 [Val]   ████████████████ 100%  loss: 1.9876 acc: 52.34%
```

### ✅ Automatic Checkpointing
- **Best model**: Saved when validation loss improves
  - Location: `models/cropshield_cnn.pth` (or `models/efficientnet.pth`)
- **Periodic**: Saved every N epochs (default: 10)
  - Location: `models/cropshield_cnn_checkpoint_epoch10.pth`
- **Final**: Saved at end of training
  - Location: `models/cropshield_cnn_final.pth`

### ✅ Training History (JSON)
```json
{
    "train_loss": [2.5, 2.1, 1.8, ...],
    "train_acc": [35.2, 45.6, 58.3, ...],
    "val_loss": [2.3, 1.9, 1.7, ...],
    "val_acc": [38.5, 48.2, 61.5, ...],
    "learning_rate": [0.0001, 0.0001, 0.00005, ...]
}
```
- Location: `training_history.json`
- Use for plotting curves

### ✅ Learning Rate Scheduling
- **StepLR**: Reduces LR by 50% every 5 epochs
  - Epoch 1-5: LR = 1e-4
  - Epoch 6-10: LR = 5e-5
  - Epoch 11-15: LR = 2.5e-5
  - ...

### ✅ Early Stopping (Optional)
- Monitors validation loss
- Stops if no improvement for N epochs
- Example: `--early_stopping 15`

---

## 📊 Training Output

### Console Output
```
============================================================
🌾 CROPSHIELD AI - TRAINING
============================================================
Model:               custom
Epochs:              25
Batch size:          32
Accumulation steps:  1
Effective batch:     32
Learning rate:       0.0001
Freeze backbone:     False
Mixed precision:     True
============================================================

📦 Loading dataset...
✅ Dataset loaded: 22 classes, 22,387 images

📦 Creating model: custom
✅ Using GPU: NVIDIA GeForce RTX 4060 Laptop GPU
   Available Memory: 8.59 GB
...

============================================================
🚀 TRAINER INITIALIZED
============================================================
Device:              cuda
Mixed Precision:     True
Accumulation Steps:  1
Effective Batch:     32
Train samples:       17909
Val samples:         2238
Test samples:        2240
============================================================

============================================================
🎯 TRAINING STARTED
============================================================
Max epochs:          25
Early stopping:      Disabled
Save every:          10 epochs
============================================================

Epoch 1 [Train] ████████████████ 100%  loss: 2.1234 acc: 45.67%
Epoch 1 [Val]   ████████████████ 100%  loss: 1.9876 acc: 52.34%

Epoch 1/25 Summary:
  Train Loss: 2.1234 | Train Acc: 45.67%
  Val Loss:   1.9876 | Val Acc:   52.34%
  LR: 0.000100 | Time: 156.2s
  🏆 New best model! Val Loss: 1.9876
------------------------------------------------------------
...

============================================================
✅ TRAINING COMPLETE
============================================================
Total time:          65.3 minutes
Best val loss:       0.4523
Best val acc:        85.67%
============================================================

✅ Best model saved: models\cropshield_cnn.pth
📊 Training history saved: training_history.json
```

### Generated Files
```
models/
├── cropshield_cnn.pth                    # Best model
├── cropshield_cnn_checkpoint_epoch10.pth # Periodic checkpoint
├── cropshield_cnn_checkpoint_epoch20.pth # Periodic checkpoint
└── cropshield_cnn_final.pth              # Final model

training_history.json                      # Metrics log
```

---

## 🔧 Troubleshooting

### CUDA Out of Memory
```bash
# Option 1: Reduce batch size
python train.py --model custom --epochs 25 --batch_size 16

# Option 2: Use gradient accumulation
python train.py --model custom --epochs 25 --batch_size 16 --accumulation_steps 2

# Option 3: Disable AMP (not recommended)
python train.py --model custom --epochs 25 --amp False
```

### Slow Training
```bash
# Increase workers (if CPU bottleneck)
python train.py --model custom --epochs 25 --num_workers 12

# Enable AMP (if not already)
python train.py --model custom --epochs 25 --amp True

# Reduce augmentation
python train.py --model custom --epochs 25 --augmentation_mode light
```

### Model Not Learning
```bash
# Try higher learning rate
python train.py --model custom --epochs 25 --lr 1e-3

# Use lighter augmentation
python train.py --model custom --epochs 25 --augmentation_mode light

# Check if labels are correct in dataset
```

### Resume After Crash
```bash
# Resume from last checkpoint
python train.py --model custom --epochs 50 --resume models/custom_checkpoint_epoch20.pth
```

---

## 📈 Expected Training Times (RTX 4060)

| Model | Epochs | Batch Size | AMP | Time |
|-------|--------|------------|-----|------|
| Custom CNN | 25 | 32 | ✅ | ~40 min |
| Custom CNN | 100 | 32 | ✅ | ~2.5 hours |
| EfficientNet-B0 | 25 | 32 | ✅ | ~30 min |
| EfficientNet-B0 | 100 | 32 | ✅ | ~2 hours |

---

## 🎯 Recommended Training Strategies

### Strategy 1: Quick Baseline (Custom CNN)
```bash
# 25 epochs, moderate augmentation
python train.py --model custom --epochs 25 --augmentation_mode moderate
```
- **Goal**: Establish baseline performance
- **Time**: ~40 minutes
- **Expected**: 70-80% accuracy

### Strategy 2: Transfer Learning (EfficientNet-B0)
```bash
# 25 epochs, frozen backbone for first few epochs
python train.py --model efficientnet --epochs 25 --freeze_backbone
```
- **Goal**: Leverage pretrained features
- **Time**: ~30 minutes
- **Expected**: 85-95% accuracy

### Strategy 3: Full Training (Best Performance)
```bash
# 100 epochs with early stopping
python train.py --model efficientnet --epochs 100 --early_stopping 15
```
- **Goal**: Maximum accuracy
- **Time**: ~1-2 hours (may stop early)
- **Expected**: 92-96% accuracy

### Strategy 4: Experiment Tracking
```bash
# Multiple runs with different configs
python train.py --model custom --epochs 25 --run_name "exp1_lr1e4"
python train.py --model custom --epochs 25 --run_name "exp2_lr1e3" --lr 1e-3
python train.py --model efficientnet --epochs 25 --run_name "exp3_frozen" --freeze_backbone
```
- **Goal**: Compare configurations
- **Result**: Separate checkpoints for each run

---

## 📚 Next Steps After Training

1. **Evaluate on Test Set**
   - Test accuracy printed at end of training
   - Use best model checkpoint

2. **Visualize Training Curves**
   - Load `training_history.json`
   - Plot loss/accuracy curves

3. **Generate Confusion Matrix**
   - Create evaluation script
   - Analyze per-class performance

4. **Implement GradCAM**
   - Visualize model attention
   - Verify focuses on disease symptoms

5. **Deploy to Streamlit**
   - Load best checkpoint
   - Real-time inference

---

**Status**: ✅ Ready to Train  
**Script**: `train.py`  
**Documentation**: Complete
