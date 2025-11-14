# Training Performance Quick Reference - CropShield AI

## 🎯 TL;DR - Ready-to-Use Commands

### Custom CNN (4.7M params) - Default
```bash
python train.py --model custom --epochs 25
```
- **Batch:** 32 (physical)
- **Effective Batch:** 32
- **AMP:** Enabled
- **VRAM:** ~6-7GB
- **Speed:** ~2.5 min/epoch

---

### EfficientNet-B0 (4.0M params) - Recommended
```bash
python train.py \
  --model efficientnet_b0 \
  --epochs 25 \
  --batch_size 16 \
  --accumulation_steps 2 \
  --scheduler plateau \
  --warmup_epochs 1
```
- **Batch:** 16 (physical)
- **Effective Batch:** 32 (16 × 2)
- **AMP:** Enabled
- **VRAM:** ~5-6GB
- **Speed:** ~1.8 min/epoch

---

### EfficientNet-B0 - Large Effective Batch (Most Stable)
```bash
python train.py \
  --model efficientnet_b0 \
  --epochs 25 \
  --batch_size 16 \
  --accumulation_steps 4 \
  --clip_grad_norm 1.0 \
  --scheduler plateau \
  --warmup_epochs 1
```
- **Batch:** 16 (physical)
- **Effective Batch:** 64 (16 × 4)
- **Gradient Clipping:** 1.0
- **VRAM:** ~5-6GB
- **Speed:** ~1.8 min/epoch

---

### Unstable Training Fix
```bash
python train.py \
  --clip_grad_norm 0.5 \
  --warmup_epochs 2 \
  --scheduler plateau \
  --scheduler_patience 5
```
- **Gradient Clipping:** 0.5 (conservative)
- **Warmup:** 2 epochs (extra stability)
- **Scheduler:** Plateau with patience=5

---

## 📊 Feature Summary

| Feature | What It Does | Benefit | Command |
|---------|--------------|---------|---------|
| **Mixed Precision (AMP)** | Uses FP16+FP32 | 2x faster, 50% less VRAM | `--amp True` (default) |
| **Gradient Accumulation** | Simulates larger batches | Stable training, less VRAM | `--accumulation_steps 2` |
| **Gradient Clipping** | Limits gradient magnitude | Prevents explosion | `--clip_grad_norm 1.0` |

---

## 🔧 Configuration Matrix

### RTX 4060 Laptop (8GB VRAM)

| Model | Physical Batch | Accumulation | Effective Batch | Clipping | VRAM | Command |
|-------|----------------|--------------|-----------------|----------|------|---------|
| **Custom CNN** | 32 | 1 | 32 | None | 6-7GB | Default |
| **Custom CNN (stable)** | 32 | 1 | 32 | 1.0 | 6-7GB | `--clip_grad_norm 1.0` |
| **EfficientNet** | 16 | 2 | 32 | None | 5-6GB | `--batch_size 16 --accumulation_steps 2` |
| **EfficientNet (stable)** | 16 | 4 | 64 | 1.0 | 5-6GB | `--batch_size 16 --accumulation_steps 4 --clip_grad_norm 1.0` |

---

### RTX 3050 Laptop (4GB VRAM)

| Model | Physical Batch | Accumulation | Effective Batch | Clipping | VRAM | Command |
|-------|----------------|--------------|-----------------|----------|------|---------|
| **Custom CNN** | 16 | 2 | 32 | 1.0 | 3.5GB | `--batch_size 16 --accumulation_steps 2 --clip_grad_norm 1.0` |
| **EfficientNet** | 8 | 4 | 32 | 1.0 | 3GB | `--batch_size 8 --accumulation_steps 4 --clip_grad_norm 1.0` |

---

## ⚡ Quick Troubleshooting

| Problem | Solution |
|---------|----------|
| **Out of Memory** | `--batch_size 16` or `--batch_size 8` |
| **Loss Spikes** | `--clip_grad_norm 1.0` |
| **NaN Loss** | `--clip_grad_norm 0.5 --warmup_epochs 1` |
| **Too Slow** | `--accumulation_steps 1` (trade stability) |
| **Unstable Validation** | `--accumulation_steps 4 --clip_grad_norm 1.0` |

---

## 📝 Parameter Guide

### --batch_size (Physical Batch Size)
- **Default:** 32 (custom), 16 (efficientnet)
- **Range:** 8-64
- **Memory Impact:** Higher = more VRAM
- **Rule:** Largest that fits in VRAM

### --accumulation_steps (Gradient Accumulation)
- **Default:** 1 (no accumulation)
- **Range:** 1-8
- **Memory Impact:** None (same VRAM as batch_size=1)
- **Rule:** `effective_batch = batch_size × accumulation_steps`
- **Formula:** `target_batch // max_physical_batch`

### --clip_grad_norm (Gradient Clipping)
- **Default:** None (disabled)
- **Range:** 0.5-5.0
- **Recommended:** 1.0 (balanced)
- **Conservative:** 0.5 (very stable, slower convergence)
- **Aggressive:** 2.0-5.0 (faster, risk instability)
- **When to use:** Loss spikes, NaN loss, unstable training

### --amp (Mixed Precision)
- **Default:** True (enabled)
- **CUDA Only:** Automatically disabled on CPU
- **Benefit:** 2x faster, 50% less VRAM
- **When to disable:** Old GPU (GTX 10xx), CPU training

---

## 🎓 Formulas

### Effective Batch Size
```python
effective_batch = batch_size × accumulation_steps
```

**Example:**
- `batch_size=16`, `accumulation_steps=4`
- **Effective batch = 64**

---

### Accumulation Steps Calculation
```python
accumulation_steps = target_batch_size // max_physical_batch_size
```

**Example:**
- Target: 64 (optimal for stable training)
- Max physical: 16 (GPU memory limit)
- **Accumulation = 64 // 16 = 4**

---

### Memory Estimation (Rough)
```python
vram_needed_gb = (
    model_size_gb +           # Model parameters
    batch_size * 0.002 +      # Input data (224x224x3)
    model_size_gb * 2 +       # Gradients + optimizer states
    2.0                       # PyTorch overhead
)
```

**Example (EfficientNet-B0, batch=16):**
- Model: 0.5 GB
- Input: 16 × 0.002 = 0.032 GB
- Gradients: 0.5 × 2 = 1.0 GB
- Overhead: 2.0 GB
- **Total ≈ 3.5 GB** ✅ Fits in 8GB VRAM

---

## 🚀 Common Workflows

### 1. First Training Run (Stable, Safe)
```bash
python train.py \
  --model custom \
  --epochs 25 \
  --batch_size 32
```
- Default settings
- No gradient clipping
- Simple, reliable

---

### 2. Transfer Learning (Recommended)
```bash
python train.py \
  --model efficientnet_b0 \
  --epochs 25 \
  --batch_size 16 \
  --accumulation_steps 2 \
  --scheduler plateau \
  --warmup_epochs 1
```
- Effective batch: 32
- Adaptive LR scheduling
- 1-epoch warmup

---

### 3. Maximum Stability (Unstable Training)
```bash
python train.py \
  --model efficientnet_b0 \
  --epochs 25 \
  --batch_size 16 \
  --accumulation_steps 4 \
  --clip_grad_norm 0.5 \
  --scheduler plateau \
  --scheduler_patience 5 \
  --warmup_epochs 2 \
  --lr 5e-5
```
- Effective batch: 64 (very stable)
- Conservative gradient clipping (0.5)
- Patient scheduler (wait 5 epochs)
- 2-epoch warmup
- Lower learning rate

---

### 4. Limited VRAM (4GB GPU)
```bash
python train.py \
  --model custom \
  --epochs 25 \
  --batch_size 16 \
  --accumulation_steps 2 \
  --clip_grad_norm 1.0
```
- Physical batch: 16 (fits in 4GB)
- Effective batch: 32 (maintained)
- Gradient clipping for safety

---

### 5. Fast Experiments (Quick Iterations)
```bash
python train.py \
  --model custom \
  --epochs 10 \
  --batch_size 32 \
  --accumulation_steps 1 \
  --scheduler step \
  --scheduler_step_size 3
```
- No accumulation (fastest)
- Aggressive scheduler (step=3)
- Short training (10 epochs)

---

## 📚 Implementation Details

### Mixed Precision (AMP)
```python
# Automatically handled in train.py
if use_amp and device.type == 'cuda':
    scaler = GradScaler()
    with autocast():
        outputs = model(images)
        loss = criterion(outputs, labels)
    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()
```

### Gradient Accumulation
```python
# Automatically handled in train.py
loss = loss / accumulation_steps  # Scale loss
loss.backward()                    # Accumulate gradients

if (batch_idx + 1) % accumulation_steps == 0:
    optimizer.step()               # Update every N batches
    optimizer.zero_grad()
```

### Gradient Clipping
```python
# Automatically handled in train.py
if clip_grad_norm is not None:
    if use_amp:
        scaler.unscale_(optimizer)  # Unscale first with AMP
    torch.nn.utils.clip_grad_norm_(
        model.parameters(), 
        max_norm=clip_grad_norm
    )
```

---

## ✅ Current Status

**All features implemented in train.py:**
- ✅ Mixed Precision (AMP) with CPU fallback
- ✅ Gradient Accumulation (1-8 steps)
- ✅ Gradient Clipping (optional)
- ✅ Automatic scaler handling
- ✅ Correct step timing
- ✅ CLI arguments for all features

**Ready to use now!** No additional setup needed.

---

## 📖 Full Documentation

See `TRAINING_PERFORMANCE_GUIDE.md` for:
- Detailed explanations
- Advanced usage
- Troubleshooting guide
- Memory estimation formulas
- Full code examples

---

*Training Performance Quick Reference - All features ready! 🚀*
