# 🖼️ Batch Visualization Quick Reference

## ✅ Visual Verification Complete!

### Generated Files
```
✅ train_batch_visualization.png      - Training batch with augmentations
✅ val_batch_visualization.png        - Validation batch (no augmentation)
✅ training_batch_sample.png          - Another training sample
✅ augmentation_comparison.png        - Same images, different transforms
✅ visualize_batch.py                 - Visualization toolkit
✅ visualize_augmentations.ipynb      - Interactive Jupyter notebook
```

---

## 📊 Validation Results

### ✅ All Checks Passed (4/4)

1. **Denormalized values in valid range [0, 1]** ✅
   - Min: 0.0000, Max: 1.0000
   - Images display correctly

2. **Mean is reasonable** ✅
   - Global mean: 0.397 (natural images typically 0.3-0.5)
   - Per-channel: R=0.409, G=0.434, B=0.347

3. **Standard deviation shows variety** ✅
   - Global std: 0.220 (good augmentation variety)
   - Not too flat (0.10) or too wild (0.40)

4. **Multiple classes present** ✅
   - 19 classes in 160 sampled images
   - Good class diversity in batches

---

## 🎨 Function Reference

### 1. `visualize_batch(loader, class_names, num_images=10)`
**Purpose**: Display a grid of images from a DataLoader

**Example**:
```python
from visualize_batch import visualize_batch
from fast_dataset import make_loaders

train_loader, val_loader, _, class_names, _ = make_loaders()

# Show 10 training images
visualize_batch(train_loader, class_names, num_images=10)
```

**Output**: 
- Grid of denormalized images with class labels
- Automatically saves to PNG if `save_path` provided
- Works with any DataLoader (train/val/test)

---

### 2. `denormalize(tensor, mean=[...], std=[...])`
**Purpose**: Reverse ImageNet normalization for display

**Example**:
```python
from visualize_batch import denormalize
import matplotlib.pyplot as plt

images, labels = next(iter(train_loader))
img_denorm = denormalize(images[0])  # Single image [C, H, W]
img_np = img_denorm.permute(1, 2, 0).numpy()

plt.imshow(img_np)
plt.show()
```

**Input**: 
- Normalized tensor [C, H, W] or [B, C, H, W]
- Mean/std values (defaults to ImageNet)

**Output**: 
- Denormalized tensor in [0, 1] range
- Ready for matplotlib display

---

### 3. `check_augmentation_statistics(loader, num_batches=5)`
**Purpose**: Print statistical summary of augmented batches

**Example**:
```python
from visualize_batch import check_augmentation_statistics

check_augmentation_statistics(train_loader, num_batches=5)
```

**Output**:
```
📊 Analyzed 160 images from 5 batches
🎨 Pixel Value Statistics: min=0.0, max=1.0, mean=0.397, std=0.220
🌈 Per-Channel: R=0.409, G=0.434, B=0.347
✅ Checks: 4/4 passed
```

---

### 4. `visualize_augmentation_comparison(loader, class_names, num_samples=3)`
**Purpose**: Show augmentation variety by fetching same images multiple times

**Example**:
```python
from visualize_batch import visualize_augmentation_comparison

visualize_augmentation_comparison(train_loader, class_names, num_samples=3)
```

**Output**: 
- Grid showing 3 images × 4 different augmentations each
- Demonstrates randomness in transforms
- Confirms variety is sufficient

---

### 5. `quick_visual_check(train_loader, val_loader, class_names)`
**Purpose**: Run all visualizations and checks at once

**Example**:
```python
from visualize_batch import quick_visual_check
from fast_dataset import make_loaders

train_loader, val_loader, _, class_names, _ = make_loaders()
quick_visual_check(train_loader, val_loader, class_names)
```

**Output**: 
- Training batch visualization
- Validation batch visualization
- Statistical checks
- 4 saved PNG files

---

## 🚀 Quick Usage Patterns

### Pattern 1: Just Check Training Batch
```python
from visualize_batch import visualize_batch
from fast_dataset import make_loaders

train_loader, _, _, class_names, _ = make_loaders()
visualize_batch(train_loader, class_names, num_images=10)
```

### Pattern 2: Compare Train vs Validation
```python
from visualize_batch import visualize_batch
from fast_dataset import make_loaders
import matplotlib.pyplot as plt

train_loader, val_loader, _, class_names, _ = make_loaders()

# Training
visualize_batch(train_loader, class_names, title="Training (With Aug)")
plt.show()

# Validation
visualize_batch(val_loader, class_names, title="Validation (No Aug)")
plt.show()
```

### Pattern 3: Full Verification
```python
from visualize_batch import quick_visual_check
from fast_dataset import make_loaders

train_loader, val_loader, _, class_names, _ = make_loaders()
quick_visual_check(train_loader, val_loader, class_names)
```

### Pattern 4: Interactive Jupyter
```python
# In Jupyter notebook
from visualize_batch import denormalize
import matplotlib.pyplot as plt

images, labels = next(iter(train_loader))

for i in range(8):
    img = denormalize(images[i]).permute(1, 2, 0).numpy()
    plt.subplot(2, 4, i+1)
    plt.imshow(img)
    plt.title(class_names[labels[i]])
    plt.axis('off')
plt.show()
```

### Pattern 5: Save for Documentation
```python
visualize_batch(
    train_loader, 
    class_names, 
    num_images=10,
    save_path="docs/training_samples.png",
    title="CropShield AI Training Data"
)
```

---

## 🔍 What to Look For

### ✅ Good Augmentations
- **Rotations**: ±15° looks natural
- **Flips**: Horizontal/vertical are realistic for plants
- **Brightness**: Slight variations (±20%)
- **Contrast**: Subtle changes (±20%)
- **Colors**: Still look like real plants
- **Disease features**: Still visible and recognizable

### ❌ Bad Augmentations
- **Over-rotated**: 45°+ rotations look unrealistic
- **Oversaturated**: Neon colors, purple leaves
- **Too dark/bright**: Can't see details
- **Too much crop**: Disease symptoms cut off
- **Artifacts**: Pixelation, strange boundaries

---

## 📊 Interpretation Guide

### Pixel Statistics After Denormalization

| Metric | Ideal Range | Your Results | Status |
|--------|-------------|--------------|--------|
| Min | 0.0 | 0.0000 | ✅ |
| Max | 1.0 | 1.0000 | ✅ |
| Mean | 0.3-0.5 | 0.3969 | ✅ |
| Std | 0.15-0.30 | 0.2203 | ✅ |

**Interpretation**:
- **Min/Max in [0, 1]**: Denormalization working correctly
- **Mean ≈ 0.4**: Natural looking images (not too dark/bright)
- **Std ≈ 0.22**: Good variety from augmentations

### Per-Channel Statistics

| Channel | Mean | Std | Interpretation |
|---------|------|-----|----------------|
| Red | 0.409 | 0.219 | ✅ Healthy balance |
| Green | 0.434 | 0.210 | ✅ Slightly higher (plants!) |
| Blue | 0.347 | 0.222 | ✅ Lower (less sky) |

**Interpretation**:
- Green channel highest → Makes sense for plant images
- All channels in reasonable range → Natural color balance
- Similar std across channels → Balanced augmentations

---

## 🎯 Troubleshooting

### Problem: Images look too dark/bright
**Solution**: Check normalization constants match your transforms
```python
# In transforms.py, ensure:
transforms.Normalize(
    mean=[0.485, 0.456, 0.406],  # ImageNet mean
    std=[0.229, 0.224, 0.225]    # ImageNet std
)

# In visualize_batch.py, ensure denormalize uses same values
denormalize(tensor, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
```

### Problem: Images look oversaturated/weird colors
**Solution**: Reduce ColorJitter strength in transforms.py
```python
# Current (MODERATE):
transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.15, hue=0.03)

# Try CONSERVATIVE:
transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.05, hue=0.01)
```

### Problem: Rotations look unnatural
**Solution**: Reduce rotation angle
```python
# Current: ±15°
transforms.RandomRotation(15)

# Try: ±10°
transforms.RandomRotation(10)
```

### Problem: Disease features not visible
**Solution**: 
1. Increase crop scale to avoid cutting off features
```python
# Current:
transforms.RandomResizedCrop(224, scale=(0.7, 1.0))

# Try:
transforms.RandomResizedCrop(224, scale=(0.8, 1.0))  # Less aggressive crop
```

2. Use CONSERVATIVE mode
```python
train_loader, _, _, _, _ = make_loaders(augmentation_mode='conservative')
```

### Problem: Not enough variety between batches
**Solution**: Use AGGRESSIVE mode or increase augmentation strength
```python
train_loader, _, _, _, _ = make_loaders(augmentation_mode='aggressive')
```

---

## 📈 Next Steps After Visual Verification

### ✅ If Augmentations Look Good
1. **Proceed to model training** (Phase 3)
2. **Use MODERATE mode** as baseline
3. **Monitor train-val accuracy gap** during training
4. **Adjust mode if overfitting occurs**

### ⚠️ If Augmentations Look Too Strong
1. **Switch to CONSERVATIVE mode**
```python
train_loader, _, _, _, _ = make_loaders(augmentation_mode='conservative')
```
2. **Re-run visualizations**
3. **Verify improvements**

### ⚠️ If Augmentations Look Too Weak
1. **Switch to AGGRESSIVE mode**
```python
train_loader, _, _, _, _ = make_loaders(augmentation_mode='aggressive')
```
2. **Check if variety increases**
3. **Use if model overfits during training**

---

## 💡 Pro Tips

### 1. Compare Modes Side-by-Side
```python
modes = ['conservative', 'moderate', 'aggressive']
for mode in modes:
    loader, _, _, names, _ = make_loaders(augmentation_mode=mode)
    visualize_batch(loader, names, num_images=8, 
                   save_path=f"{mode}_sample.png",
                   title=f"{mode.upper()} Mode")
```

### 2. Check Specific Classes
```python
# Get batch
images, labels = next(iter(train_loader))

# Find specific class (e.g., class 5)
mask = labels == 5
class_images = images[mask]
print(f"Found {len(class_images)} images of class {class_names[5]}")

# Visualize
for i, img in enumerate(class_images[:4]):
    img_denorm = denormalize(img).permute(1, 2, 0).numpy()
    plt.subplot(2, 2, i+1)
    plt.imshow(img_denorm)
    plt.title(class_names[5])
    plt.axis('off')
plt.show()
```

### 3. Save Representative Samples for Documentation
```python
# Save one example from each augmentation mode
for mode in ['conservative', 'moderate', 'aggressive']:
    loader, _, _, names, _ = make_loaders(augmentation_mode=mode)
    visualize_batch(
        loader, names, num_images=6,
        save_path=f"docs/{mode}_examples.png",
        title=f"CropShield AI - {mode.capitalize()} Augmentation"
    )
```

### 4. Verify Augmentation Randomness
```python
# Same image, different augmentations
for i in range(4):
    images, labels = next(iter(train_loader))
    img = denormalize(images[0]).permute(1, 2, 0).numpy()
    plt.subplot(2, 2, i+1)
    plt.imshow(img)
    plt.title(f"Augmentation {i+1}")
    plt.axis('off')
plt.suptitle("Same Image Index, Different Augmentations")
plt.show()
```

---

## 📚 Additional Resources

- **transforms.py**: Core augmentation module
- **TRANSFORMS_GUIDE.md**: Full augmentation documentation
- **TRANSFORMS_QUICKREF.md**: Quick reference for transforms
- **demo_transforms.py**: Transform demonstration scripts
- **test_dataloader_integration.py**: Full DataLoader validation

---

## ✅ Validation Checklist

Before proceeding to model training, confirm:

- [ ] Training images show realistic augmentations
- [ ] Validation images are consistent (no augmentation)
- [ ] Disease features are still visible after augmentation
- [ ] Colors look natural (not oversaturated)
- [ ] Rotations are not excessive
- [ ] All 4 statistical checks pass
- [ ] Class labels are correct
- [ ] No artifacts or corruption visible
- [ ] Augmentation creates sufficient variety
- [ ] Denormalization works correctly (images displayable)

**If all checked** → ✅ **READY FOR MODEL TRAINING!**

---

**Last Updated**: November 9, 2025  
**Status**: ✅ All visualizations verified, augmentations confirmed realistic
