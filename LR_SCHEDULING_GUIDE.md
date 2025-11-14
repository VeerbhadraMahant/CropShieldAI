# Learning Rate Scheduling Guide - CropShield AI

## 📚 Table of Contents
1. [Overview](#overview)
2. [Available Schedulers](#available-schedulers)
3. [Scheduler Selection Guide](#scheduler-selection-guide)
4. [Hyperparameter Recommendations](#hyperparameter-recommendations)
5. [Usage Examples](#usage-examples)
6. [Technical Details](#technical-details)
7. [Troubleshooting](#troubleshooting)

---

## Overview

Learning rate scheduling is crucial for stable convergence and achieving high accuracy. This guide covers:
- **3 scheduler types**: StepLR, ReduceLROnPlateau, CosineAnnealingLR
- **Linear warmup**: Smooth LR ramp-up for pretrained models
- **Correct step() placement**: Per-epoch scheduling with validation loss support
- **Laptop GPU optimization**: Tested on RTX 4060 (8GB VRAM)

---

## Available Schedulers

### 1. StepLR (Default)
**Best for**: Custom CNN, stable training, predictable convergence

Reduces learning rate by a fixed factor (gamma) every N epochs (step_size).

```python
# Example: LR = 1e-4
# Epoch  0-4:  LR = 1e-4
# Epoch  5-9:  LR = 5e-5  (reduced by 0.5)
# Epoch 10-14: LR = 2.5e-5 (reduced by 0.5 again)
```

**Parameters:**
- `step_size`: Epochs between reductions (default: 5)
- `gamma`: Reduction factor (default: 0.5 = 50% reduction)

**Pros:**
- Predictable, stable convergence
- Works well for custom CNN from scratch
- No validation dependency

**Cons:**
- Not adaptive to training dynamics
- May reduce LR too early or too late

---

### 2. ReduceLROnPlateau
**Best for**: Transfer learning (EfficientNet), adaptive training, unstable validation

Reduces learning rate when validation loss stops improving (plateaus).

```python
# Example: Patience = 3, Factor = 0.5
# If validation loss doesn't improve for 3 consecutive epochs:
# LR = LR * 0.5
```

**Parameters:**
- `patience`: Epochs to wait before reducing (default: 3)
- `factor`: Reduction factor (default: 0.5)
- `mode`: 'min' (monitors val_loss, lower is better)

**Pros:**
- Adaptive to actual training progress
- Excellent for transfer learning
- Prevents premature LR reduction

**Cons:**
- Requires validation loss
- Can be conservative (slow to react)
- Adds complexity to training loop

---

### 3. CosineAnnealingLR
**Best for**: Long training runs, smooth LR schedules, advanced optimization

Anneals learning rate following a cosine curve from base LR to near zero.

```python
# Example: T_max = 50, LR = 1e-4
# Epoch  0: LR = 1e-4
# Epoch 25: LR ≈ 5e-5 (midpoint)
# Epoch 50: LR ≈ 1e-6 (minimum)
```

**Parameters:**
- `T_max`: Epochs for one complete cycle (default: 50)
- `eta_min`: Minimum LR at end of cycle (default: 1e-6)

**Pros:**
- Smooth, continuous LR decay
- Good for long training runs
- Helps escape local minima

**Cons:**
- Requires knowing total epochs in advance
- Not suitable for early stopping
- Less interpretable than StepLR

---

### 4. Linear Warmup (Optional)
**Best for**: Pretrained models (EfficientNet), preventing gradient explosion

Gradually increases learning rate from 0 to base LR over N epochs.

```python
# Example: Base LR = 1e-4, Warmup = 1 epoch
# Epoch 0: LR = 0      → 1e-4 (linear ramp)
# Epoch 1+: Use underlying scheduler (StepLR, Plateau, etc.)
```

**Parameters:**
- `warmup_epochs`: Number of warmup epochs (default: 0 = disabled)

**Pros:**
- Prevents gradient explosion in early training
- Essential for pretrained models with frozen layers
- Stabilizes training from start

**Cons:**
- Adds 1-3 epochs to training time
- Not needed for custom CNN from scratch

---

## Scheduler Selection Guide

| Scenario | Recommended Scheduler | Warmup? | Rationale |
|----------|----------------------|---------|-----------|
| **Custom CNN from scratch** | StepLR | No | Stable, predictable, no warmup needed |
| **EfficientNet-B0 (pretrained)** | ReduceLROnPlateau | Yes (1 epoch) | Adaptive + warmup prevents instability |
| **Transfer learning (any)** | ReduceLROnPlateau | Yes (1 epoch) | Adaptive to fine-tuning dynamics |
| **Long training (50+ epochs)** | CosineAnnealingLR | Optional | Smooth decay over full schedule |
| **Unstable validation loss** | ReduceLROnPlateau | Yes (1-2 epochs) | Adaptive + warmup stabilizes |
| **Fast experimentation** | StepLR | No | Simple, reliable, fast to iterate |

---

## Hyperparameter Recommendations

### 🔧 Laptop GPU Constraints (RTX 4060, 8GB VRAM)

#### Custom CNN (4.7M params)
```bash
python train.py \
  --model custom \
  --epochs 25 \
  --batch_size 32 \
  --lr 1e-4 \
  --weight_decay 1e-4 \
  --scheduler step \
  --scheduler_step_size 5 \
  --scheduler_gamma 0.5 \
  --warmup_epochs 0
```

**Why these values?**
- **Batch 32**: Fits comfortably in 8GB VRAM
- **LR 1e-4**: Balanced for Adam optimizer
- **StepLR(5, 0.5)**: Reduce LR at epochs 5, 10, 15, 20
- **No warmup**: Custom CNN trained from scratch

**Expected timeline:**
- 25 epochs: ~1 hour
- Target accuracy: 75-85%

---

#### EfficientNet-B0 (4.0M params, pretrained)
```bash
python train.py \
  --model efficientnet_b0 \
  --epochs 25 \
  --batch_size 16 \
  --lr 1e-4 \
  --weight_decay 1e-4 \
  --scheduler plateau \
  --scheduler_patience 3 \
  --scheduler_gamma 0.5 \
  --warmup_epochs 1
```

**Why these values?**
- **Batch 16**: Safer for pretrained model + augmentation
- **LR 1e-4**: Conservative for fine-tuning
- **ReduceLROnPlateau(3, 0.5)**: Adaptive to validation progress
- **Warmup 1 epoch**: Prevents gradient explosion from pretrained weights

**Expected timeline:**
- 25 epochs: ~45 minutes
- Target accuracy: 92-96%

---

### 📊 Scheduler Parameter Tuning

#### StepLR
| Parameter | Conservative | Balanced | Aggressive |
|-----------|--------------|----------|------------|
| `step_size` | 7 | 5 | 3 |
| `gamma` | 0.3 | 0.5 | 0.7 |
| **Effect** | Slower decay, stable | Good default | Faster decay, risk underfitting |

**Example:**
```bash
# Conservative (longer stable LR periods)
python train.py --scheduler step --scheduler_step_size 7 --scheduler_gamma 0.3

# Aggressive (frequent LR drops)
python train.py --scheduler step --scheduler_step_size 3 --scheduler_gamma 0.7
```

---

#### ReduceLROnPlateau
| Parameter | Conservative | Balanced | Aggressive |
|-----------|--------------|----------|------------|
| `patience` | 5 | 3 | 2 |
| `factor` | 0.3 | 0.5 | 0.7 |
| **Effect** | Wait longer, smaller drops | Good default | React fast, bigger drops |

**Example:**
```bash
# Conservative (patient, small reductions)
python train.py --scheduler plateau --scheduler_patience 5 --scheduler_gamma 0.3

# Aggressive (react quickly, large reductions)
python train.py --scheduler plateau --scheduler_patience 2 --scheduler_gamma 0.7
```

---

#### Warmup
| Warmup Epochs | Use Case |
|---------------|----------|
| 0 | Custom CNN from scratch, stable training |
| 1 | Pretrained models (EfficientNet), standard fine-tuning |
| 2 | Large batch sizes, unstable early training |
| 3 | Very deep networks, extreme batch sizes |

---

## Usage Examples

### Example 1: Basic Training (Custom CNN, Default Settings)
```bash
python train.py --model custom --epochs 25
```

**What happens:**
- StepLR scheduler (default)
- LR reduced at epochs 5, 10, 15, 20
- No warmup (not needed for custom CNN)
- Batch size 32, LR 1e-4

---

### Example 2: Transfer Learning (EfficientNet, Recommended)
```bash
python train.py \
  --model efficientnet_b0 \
  --epochs 25 \
  --batch_size 16 \
  --scheduler plateau \
  --scheduler_patience 3 \
  --warmup_epochs 1
```

**What happens:**
- Warmup epoch 0: LR ramps 0 → 1e-4
- Epochs 1+: ReduceLROnPlateau monitors val_loss
- LR reduced when val_loss plateaus for 3 epochs
- Adaptive to actual training progress

---

### Example 3: Long Training with Cosine Annealing
```bash
python train.py \
  --model custom \
  --epochs 50 \
  --scheduler cosine \
  --scheduler_t_max 50
```

**What happens:**
- LR follows cosine curve: 1e-4 → 1e-6 over 50 epochs
- Smooth, continuous decay
- No abrupt LR drops
- Good for finding optimal minima

---

### Example 4: Aggressive StepLR (Fast Experiments)
```bash
python train.py \
  --model custom \
  --epochs 25 \
  --scheduler step \
  --scheduler_step_size 3 \
  --scheduler_gamma 0.7
```

**What happens:**
- LR reduced every 3 epochs
- Larger reduction factor (0.7 = 70% → 30% LR)
- Faster convergence, but risk of underfitting
- Good for quick experiments

---

### Example 5: Conservative ReduceLROnPlateau (Stable Training)
```bash
python train.py \
  --model efficientnet_b0 \
  --epochs 30 \
  --scheduler plateau \
  --scheduler_patience 5 \
  --scheduler_gamma 0.3 \
  --warmup_epochs 2
```

**What happens:**
- 2-epoch warmup for extra stability
- Wait 5 epochs before reducing LR
- Small reduction factor (0.3 = 30% LR drop)
- Very stable, patient training

---

### Example 6: Custom Hyperparameters
```bash
python train.py \
  --model custom \
  --epochs 25 \
  --batch_size 64 \
  --lr 5e-4 \
  --weight_decay 5e-5 \
  --scheduler step \
  --scheduler_step_size 7 \
  --scheduler_gamma 0.3 \
  --accumulation_steps 2
```

**What happens:**
- Higher LR (5e-4) for faster convergence
- Larger batch size (64) with gradient accumulation
- Conservative scheduler (step=7, gamma=0.3)
- Effective batch size: 64 × 2 = 128

---

## Technical Details

### Scheduler.step() Placement

**Critical**: Different schedulers require different `step()` usage!

#### StepLR (Per-Epoch, No Arguments)
```python
for epoch in range(max_epochs):
    train_loss = train_one_epoch()
    val_loss = validate()
    
    scheduler.step()  # Call after validation, no arguments
    current_lr = optimizer.param_groups[0]['lr']
```

---

#### ReduceLROnPlateau (Per-Epoch, Pass Validation Loss)
```python
for epoch in range(max_epochs):
    train_loss = train_one_epoch()
    val_loss = validate()
    
    scheduler.step(val_loss)  # MUST pass validation loss
    current_lr = optimizer.param_groups[0]['lr']
```

**Why?** ReduceLROnPlateau monitors validation loss to detect plateaus.

---

#### CosineAnnealingLR (Per-Epoch, No Arguments)
```python
for epoch in range(max_epochs):
    train_loss = train_one_epoch()
    val_loss = validate()
    
    scheduler.step()  # Call after validation, no arguments
    current_lr = optimizer.param_groups[0]['lr']
```

---

#### Warmup + Underlying Scheduler (Automatic Handling)
```python
# The LRSchedulerWithWarmup wrapper handles both cases automatically
for epoch in range(max_epochs):
    train_loss = train_one_epoch()
    val_loss = validate()
    
    # Warmup wrapper detects if underlying scheduler is ReduceLROnPlateau
    if isinstance(scheduler, LRSchedulerWithWarmup):
        if isinstance(scheduler.scheduler, ReduceLROnPlateau):
            scheduler.step(val_loss)  # Pass val_loss for plateau
        else:
            scheduler.step()  # No arguments for StepLR/Cosine
    else:
        scheduler.step()  # Standard scheduler
```

**Note**: The `train.py` script handles this automatically! You don't need to modify code.

---

### Implementation Architecture

#### 1. LRSchedulerWithWarmup Class
Located in: `models/model_factory.py`

```python
class LRSchedulerWithWarmup:
    """
    Wrapper for linear warmup + any PyTorch scheduler.
    
    Features:
    - Linear ramp: 0 → base_lr over warmup_epochs
    - Automatic delegation to underlying scheduler
    - Supports ReduceLROnPlateau (passes metrics)
    - Compatible with StepLR, CosineAnnealingLR, etc.
    """
    
    def step(self, metrics=None):
        if self.current_epoch < self.warmup_epochs:
            # Linear warmup
            lr = self.base_lr * (self.current_epoch + 1) / self.warmup_epochs
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = lr
        else:
            # Delegate to underlying scheduler
            if isinstance(self.scheduler, ReduceLROnPlateau):
                self.scheduler.step(metrics)  # Pass val_loss
            else:
                self.scheduler.step()  # No arguments
        
        self.current_epoch += 1
```

---

#### 2. Model Factory Integration
Located in: `models/model_factory.py`

```python
def get_model(
    scheduler_type='step',
    warmup_epochs=0,
    ...
):
    # Create base scheduler
    if scheduler_type == 'step':
        base_scheduler = StepLR(...)
    elif scheduler_type == 'plateau':
        base_scheduler = ReduceLROnPlateau(...)
    elif scheduler_type == 'cosine':
        base_scheduler = CosineAnnealingLR(...)
    
    # Wrap with warmup if requested
    if warmup_epochs > 0:
        scheduler = LRSchedulerWithWarmup(optimizer, base_scheduler, warmup_epochs, lr)
    else:
        scheduler = base_scheduler
    
    return model, optimizer, criterion, scheduler, device
```

---

#### 3. Trainer Class Automatic Handling
Located in: `train.py`

```python
# In Trainer.train() method:
for epoch in range(max_epochs):
    train_loss = train_one_epoch()
    val_loss = validate()
    
    # Automatic scheduler handling
    if isinstance(scheduler, ReduceLROnPlateau):
        scheduler.step(val_loss)  # Pass val_loss
    elif hasattr(scheduler, 'step'):
        # Check if warmup wrapper with plateau inside
        if hasattr(scheduler, 'scheduler') and isinstance(scheduler.scheduler, ReduceLROnPlateau):
            scheduler.step(val_loss)  # Pass val_loss to warmup wrapper
        else:
            scheduler.step()  # Standard step
```

**Result**: All schedulers handled correctly, no manual intervention needed!

---

## Troubleshooting

### Problem 1: Training Unstable (Loss Diverges Early)
**Symptoms:**
- Loss increases dramatically in first few epochs
- NaN gradients or loss
- Model converges then suddenly diverges

**Solutions:**
1. **Add warmup** (especially for pretrained models)
   ```bash
   python train.py --warmup_epochs 1  # or 2
   ```

2. **Lower learning rate**
   ```bash
   python train.py --lr 5e-5  # Half the default
   ```

3. **Use smaller batch size**
   ```bash
   python train.py --batch_size 16  # Reduce from 32
   ```

4. **Switch to ReduceLROnPlateau** (more conservative)
   ```bash
   python train.py --scheduler plateau --scheduler_patience 5
   ```

---

### Problem 2: Training Plateaus Too Early
**Symptoms:**
- Validation accuracy stagnates around 60-70%
- Training loss still decreasing but val loss flat
- Model not learning complex patterns

**Solutions:**
1. **Slower LR decay** (increase step_size)
   ```bash
   python train.py --scheduler_step_size 7  # From 5
   ```

2. **Smaller LR reduction** (lower gamma)
   ```bash
   python train.py --scheduler_gamma 0.3  # From 0.5
   ```

3. **Switch to ReduceLROnPlateau** (adaptive)
   ```bash
   python train.py --scheduler plateau --scheduler_patience 3
   ```

4. **Train longer** (more epochs)
   ```bash
   python train.py --epochs 40  # From 25
   ```

---

### Problem 3: Validation Accuracy Oscillates
**Symptoms:**
- Val accuracy: 85% → 78% → 83% → 76%
- Loss increases after LR reduction
- Model performance unstable

**Solutions:**
1. **Add warmup** (stabilizes early training)
   ```bash
   python train.py --warmup_epochs 1
   ```

2. **Use ReduceLROnPlateau** (adapts to instability)
   ```bash
   python train.py --scheduler plateau --scheduler_patience 4
   ```

3. **Lower learning rate** (less aggressive updates)
   ```bash
   python train.py --lr 5e-5
   ```

4. **Increase patience** (wait longer before reducing LR)
   ```bash
   python train.py --scheduler plateau --scheduler_patience 5
   ```

---

### Problem 4: Custom CNN Not Learning
**Symptoms:**
- Accuracy stuck at ~20-30% (random guessing)
- Loss decreasing very slowly
- Training takes very long

**Solutions:**
1. **Use default StepLR** (simple, stable)
   ```bash
   python train.py --model custom --scheduler step
   ```

2. **Higher learning rate** (custom CNN from scratch needs faster updates)
   ```bash
   python train.py --lr 5e-4  # 5x default
   ```

3. **No warmup** (not needed for custom CNN)
   ```bash
   python train.py --warmup_epochs 0  # Ensure warmup disabled
   ```

4. **More aggressive scheduler**
   ```bash
   python train.py --scheduler_step_size 3 --scheduler_gamma 0.7
   ```

---

### Problem 5: EfficientNet Underfitting
**Symptoms:**
- Accuracy stuck at 70-80% (below expected 92-96%)
- Training loss still high
- Model not leveraging pretrained knowledge

**Solutions:**
1. **Use ReduceLROnPlateau + Warmup** (recommended for transfer learning)
   ```bash
   python train.py --model efficientnet_b0 \
     --scheduler plateau \
     --scheduler_patience 3 \
     --warmup_epochs 1
   ```

2. **Lower learning rate** (fine-tuning needs gentler updates)
   ```bash
   python train.py --lr 5e-5
   ```

3. **Train longer** (transfer learning can take more epochs)
   ```bash
   python train.py --epochs 40
   ```

4. **Don't freeze backbone** (unless explicitly needed)
   ```bash
   # Do NOT use --freeze_backbone unless you have limited data
   python train.py --model efficientnet_b0
   ```

---

### Problem 6: Out of Memory (OOM) Errors
**Symptoms:**
- CUDA out of memory error
- Training crashes during forward/backward pass
- GPU memory maxed out

**Solutions:**
1. **Reduce batch size**
   ```bash
   python train.py --batch_size 16  # From 32
   ```

2. **Use gradient accumulation** (simulate larger batches)
   ```bash
   python train.py --batch_size 16 --accumulation_steps 2  # Effective batch: 32
   ```

3. **Disable AMP** (if causing issues, though rare)
   ```bash
   python train.py --amp False
   ```

4. **Reduce num_workers** (free system RAM)
   ```bash
   python train.py --num_workers 4  # From 8
   ```

---

## Quick Reference

### Command Cheat Sheet

```bash
# 1. Custom CNN (default, recommended)
python train.py --model custom --epochs 25

# 2. EfficientNet (transfer learning, recommended)
python train.py --model efficientnet_b0 --epochs 25 --batch_size 16 \
  --scheduler plateau --warmup_epochs 1

# 3. Aggressive training (fast experiments)
python train.py --scheduler step --scheduler_step_size 3 --scheduler_gamma 0.7

# 4. Conservative training (stable, patient)
python train.py --scheduler plateau --scheduler_patience 5 --scheduler_gamma 0.3

# 5. Long training with cosine
python train.py --epochs 50 --scheduler cosine --scheduler_t_max 50

# 6. Unstable training? Add warmup + plateau
python train.py --scheduler plateau --warmup_epochs 2 --scheduler_patience 4

# 7. Resume with different scheduler
python train.py --resume models/checkpoint_epoch10.pth --scheduler plateau

# 8. Custom hyperparameters
python train.py --lr 5e-4 --weight_decay 5e-5 --batch_size 64 \
  --scheduler step --scheduler_step_size 7 --scheduler_gamma 0.3
```

---

### Scheduler Parameters Quick Reference

| Scheduler | Key Parameters | Default | Range |
|-----------|---------------|---------|-------|
| **StepLR** | `step_size` | 5 | 3-10 epochs |
|  | `gamma` | 0.5 | 0.1-0.9 |
| **ReduceLROnPlateau** | `patience` | 3 | 2-7 epochs |
|  | `factor` | 0.5 | 0.1-0.9 |
| **CosineAnnealingLR** | `T_max` | 50 | match total epochs |
|  | `eta_min` | 1e-6 | 1e-7 to 1e-5 |
| **Warmup** | `warmup_epochs` | 0 | 0-3 epochs |

---

## Next Steps

1. **Start with defaults** (StepLR for custom, plateau for EfficientNet)
2. **Monitor training** (use TensorBoard or training logs)
3. **Adjust if needed** (use troubleshooting guide)
4. **Experiment** (try different schedulers and compare results)

**Good luck training! 🚀**

---

*Last updated: Phase 6 - Learning Rate Scheduling Optimization*
