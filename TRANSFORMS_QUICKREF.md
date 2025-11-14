# Transform Quick Reference Card

## Import & Basic Usage
```python
from transforms import get_transforms

# Get transforms (choose mode)
train_tfm, val_tfm = get_transforms('moderate')  # RECOMMENDED

# Apply to PIL image
from PIL import Image
image = Image.open('sample.jpg')
tensor = train_tfm(image)  # → [3, 224, 224], float32, normalized
```

## Three Modes

| Mode | Augmentations | When to Use |
|------|--------------|-------------|
| **conservative** | 3 | First run, small dataset (<5K) |
| **moderate** | 6 | **DEFAULT** - most cases |
| **aggressive** | 8 | Large dataset (>20K), overfitting |

## Integration with DataLoaders

### FastImageFolder
```python
from fast_dataset import make_loaders

train_loader, val_loader, test_loader, _, _ = make_loaders(
    data_dir='Database_resized/',
    batch_size=32,
    augmentation_mode='moderate'
)
```

### Custom Dataset
```python
from transforms import get_transforms

train_tfm, val_tfm = get_transforms('moderate')

train_dataset = MyDataset(root='train/', transform=train_tfm)
val_dataset = MyDataset(root='val/', transform=val_tfm)
```

## Augmentation Details

### Conservative (3 augmentations)
```
RandomResizedCrop(224, scale=(0.8, 1.0))
RandomHorizontalFlip(p=0.5)
ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.02)
```

### Moderate (6 augmentations) ⭐ RECOMMENDED
```
RandomResizedCrop(224, scale=(0.7, 1.0))
RandomHorizontalFlip(p=0.5)
RandomVerticalFlip(p=0.3)
RandomRotation(±15°)
ColorJitter(brightness=0.2, contrast=0.2, saturation=0.15, hue=0.03)
RandomErasing(p=0.1)
```

### Aggressive (8 augmentations)
```
RandomResizedCrop(224, scale=(0.6, 1.0))
RandomHorizontalFlip(p=0.5)
RandomVerticalFlip(p=0.5)
RandomRotation(±20°)
ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, hue=0.05)
GaussianBlur(kernel_size=5)
RandomErasing(p=0.2)
```

## Normalization

**All modes use ImageNet stats:**
```python
mean = [0.485, 0.456, 0.406]  # RGB
std = [0.229, 0.224, 0.225]   # RGB
```

Required for transfer learning with pretrained models (ResNet, EfficientNet, etc.)

## Denormalization (for visualization)

```python
from transforms import denormalize

normalized = train_tfm(image)
original = denormalize(normalized)

import matplotlib.pyplot as plt
plt.imshow(original.permute(1, 2, 0))
```

## Validation Transforms

**No augmentation** - only preprocessing:
```python
Resize(256)
CenterCrop(224)
ToTensor()
Normalize(ImageNet)
```

## Decision Tree

```
Start with MODERATE
        ↓
Train for 10 epochs
        ↓
Monitor train-val accuracy gap
        ↓
    ┌───┴───┐
Gap > 15%  Gap < 5%
    ↓          ↓
AGGRESSIVE  ├─ High acc? → KEEP MODERATE
            └─ Low acc?  → CONSERVATIVE
```

## Performance Impact

| Mode | Relative Speed | Throughput (img/s) |
|------|---------------|-------------------|
| None | 1.00x | 1,096 |
| Conservative | 0.91x | 995 |
| Moderate | 0.85x | 931 |
| Aggressive | 0.75x | 823 |

*Hidden by GPU training time with DataLoader prefetching*

## Troubleshooting

### Images look unrealistic
```python
# Reduce augmentation strength
train_tfm, val_tfm = get_transforms('conservative')
```

### Overfitting (train >> val accuracy)
```python
# Increase augmentation strength
train_tfm, val_tfm = get_transforms('aggressive')
```

### Underfitting (both accuracies low)
```python
# Decrease augmentation strength
train_tfm, val_tfm = get_transforms('conservative')
# Or check model capacity, learning rate
```

### CUDA out of memory
```python
# Reduce batch size (not related to transforms)
train_loader = DataLoader(..., batch_size=16)  # was 32
```

## Testing

### Visual Check
```python
from transforms import visualize_augmentations

visualize_augmentations(
    'sample.jpg',
    mode='moderate',
    num_samples=8,
    save_path='augmentation_demo.png'
)
```

### Demo Script
```bash
python demo_transforms.py
# Generates visual demos for 3 sample images
```

## Files

| File | Purpose |
|------|---------|
| `transforms.py` | Core module |
| `demo_transforms.py` | Visualization tool |
| `TRANSFORMS_GUIDE.md` | Full documentation |
| `TRANSFORMS_QUICKREF.md` | This card |

## Common Patterns

### Training Loop
```python
from transforms import get_transforms

train_tfm, val_tfm = get_transforms('moderate')

# Training
model.train()
for images, labels in train_loader:
    # images already augmented by train_tfm
    outputs = model(images.to(device))
    loss = criterion(outputs, labels.to(device))
    # ...

# Validation
model.eval()
with torch.no_grad():
    for images, labels in val_loader:
        # images only normalized by val_tfm
        outputs = model(images.to(device))
        # ...
```

### Custom Normalization
```python
train_tfm, val_tfm = get_transforms(
    mode='moderate',
    mean=[0.5, 0.5, 0.5],  # Your dataset mean
    std=[0.2, 0.2, 0.2]    # Your dataset std
)
```

---

**Remember**: Start with **MODERATE**, monitor metrics, adjust as needed!
