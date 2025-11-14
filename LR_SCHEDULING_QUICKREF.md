# LR Scheduling Quick Reference - CropShield AI

## 🎯 Recommended Configurations

### Custom CNN (From Scratch)
```bash
python train.py \
  --model custom \
  --epochs 25 \
  --batch_size 32 \
  --lr 1e-4 \
  --scheduler step \
  --scheduler_step_size 5 \
  --scheduler_gamma 0.5 \
  --warmup_epochs 0
```
**Why**: Simple, stable, no warmup needed. LR reduced at epochs 5, 10, 15, 20.

---

### EfficientNet-B0 (Transfer Learning)
```bash
python train.py \
  --model efficientnet_b0 \
  --epochs 25 \
  --batch_size 16 \
  --lr 1e-4 \
  --scheduler plateau \
  --scheduler_patience 3 \
  --scheduler_gamma 0.5 \
  --warmup_epochs 1
```
**Why**: Adaptive scheduling + warmup prevents instability. LR adapts to validation loss.

---

## 📊 Scheduler Comparison

| Scheduler | When to Use | Pros | Cons |
|-----------|-------------|------|------|
| **StepLR** | Custom CNN, stable training | Predictable, simple | Not adaptive |
| **ReduceLROnPlateau** | Transfer learning, unstable val | Adaptive, smart | Requires val_loss |
| **CosineAnnealingLR** | Long training (50+ epochs) | Smooth decay | Needs total epochs |
| **Warmup** | Pretrained models | Prevents instability | Adds 1-3 epochs |

---

## 🔧 Quick Troubleshooting

| Problem | Solution |
|---------|----------|
| **Training unstable** | Add warmup: `--warmup_epochs 1` |
| **Plateaus early** | Increase step_size: `--scheduler_step_size 7` |
| **Val accuracy oscillates** | Use plateau: `--scheduler plateau --scheduler_patience 4` |
| **Custom CNN not learning** | Higher LR: `--lr 5e-4`, use StepLR |
| **EfficientNet underfitting** | Use plateau + warmup, train 40 epochs |
| **Out of memory** | Reduce batch: `--batch_size 16` |

---

## 📝 Common Commands

```bash
# Default (Custom CNN)
python train.py --model custom --epochs 25

# Transfer Learning (Recommended)
python train.py --model efficientnet_b0 --epochs 25 --batch_size 16 \
  --scheduler plateau --warmup_epochs 1

# Fast Experiments (Aggressive)
python train.py --scheduler step --scheduler_step_size 3 --scheduler_gamma 0.7

# Stable Training (Conservative)
python train.py --scheduler plateau --scheduler_patience 5 --scheduler_gamma 0.3

# Long Training (Cosine)
python train.py --epochs 50 --scheduler cosine --scheduler_t_max 50

# Resume with Different Scheduler
python train.py --resume models/checkpoint_epoch10.pth --scheduler plateau
```

---

## 🎓 Parameter Ranges

| Parameter | Default | Conservative | Balanced | Aggressive |
|-----------|---------|--------------|----------|------------|
| **step_size** | 5 | 7 | 5 | 3 |
| **gamma** | 0.5 | 0.3 | 0.5 | 0.7 |
| **patience** | 3 | 5 | 3 | 2 |
| **warmup_epochs** | 0 | 2 | 1 | 0 |

---

## ⚡ Laptop GPU Settings (RTX 4060, 8GB)

| Model | Batch Size | LR | Scheduler | Warmup | Time/Epoch |
|-------|------------|----|-----------|---------|-----------| 
| **Custom CNN** | 32 | 1e-4 | StepLR | 0 | ~2.5 min |
| **EfficientNet-B0** | 16 | 1e-4 | Plateau | 1 | ~1.8 min |

---

## 📚 Full Documentation

See `LR_SCHEDULING_GUIDE.md` for:
- Detailed scheduler explanations
- Hyperparameter tuning guide
- Technical implementation details
- Extensive troubleshooting
- Usage examples

---

*Quick reference for Phase 6 - Learning Rate Scheduling*
