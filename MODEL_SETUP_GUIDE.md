# Model Setup Guide for CropShield AI

## Overview
Complete guide for initializing and setting up models (custom CNN or transfer learning) with recommended hyperparameters.

## Quick Start

### One-Line Setup (Recommended)
```python
from model_setup import setup_model_for_training
from model_custom_cnn import CropShieldCNN

# Create model
model = CropShieldCNN(num_classes=22)

# Complete setup
setup = setup_model_for_training(model)

# Extract components
model = setup['model']
device = setup['device']
criterion = setup['criterion']
optimizer = setup['optimizer']
scheduler = setup['scheduler']
```

## Recommended Hyperparameters

### Custom CNN (Training from Scratch)
```python
setup = setup_model_for_training(
    model=model,
    learning_rate=1e-4,      # Conservative LR
    weight_decay=1e-4,       # L2 regularization
    step_size=5,             # Reduce LR every 5 epochs
    gamma=0.5,               # Reduce by 50%
    optimizer_type='adam'    # Adam optimizer
)
```

### Transfer Learning (Fine-tuning)
```python
import torchvision.models as models

# Load pretrained model
model = models.efficientnet_b0(weights='IMAGENET1K_V1')
model.classifier[1] = nn.Linear(1280, 22)

# Setup with same hyperparameters
setup = setup_model_for_training(
    model=model,
    learning_rate=1e-4,      # Lower LR for fine-tuning
    weight_decay=1e-4,
    step_size=5,
    gamma=0.5,
    optimizer_type='adam'
)
```

## Available Functions

### 1. `get_device()`
Auto-detect best available device (CUDA or CPU).

```python
from model_setup import get_device

device = get_device()
# ✅ Using GPU: NVIDIA GeForce RTX 4060 Laptop GPU
#    Memory Available: 8.59 GB
```

### 2. `count_parameters(model, verbose=True)`
Count model parameters (total, trainable, non-trainable).

```python
from model_setup import count_parameters

param_info = count_parameters(model)
# Returns: {'total': 4698582, 'trainable': 4698582, 'non_trainable': 0}
```

### 3. `get_model_summary(model, input_size=(3,224,224))`
Comprehensive model summary with memory usage.

```python
from model_setup import get_model_summary

summary = get_model_summary(model)
# Prints detailed summary and returns:
# {
#   'total_params': 4698582,
#   'trainable_params': 4698582,
#   'non_trainable_params': 0,
#   'param_memory_mb': 17.92,
#   'activation_memory_mb': 155.04,
#   'total_memory_mb': 172.97,
#   'input_size': (3, 224, 224),
#   'device': 'cuda'
# }
```

### 4. `initialize_model(model, device=None)`
Move model to device and print summary.

```python
from model_setup import initialize_model

model, device = initialize_model(model)
# ✅ Model moved to cuda
# 📊 Model size: 17.92 MB
# 💾 Available GPU memory: 8.59 GB
```

### 5. `create_training_setup(model, learning_rate=1e-4, ...)`
Create loss function, optimizer, and scheduler.

```python
from model_setup import create_training_setup

model, criterion, optimizer, scheduler = create_training_setup(
    model=model,
    learning_rate=1e-4,
    weight_decay=1e-4,
    step_size=5,
    gamma=0.5,
    optimizer_type='adam'
)
```

### 6. `setup_model_for_training(model, ...)` [RECOMMENDED]
Complete one-stop setup function.

```python
from model_setup import setup_model_for_training

setup = setup_model_for_training(
    model=model,
    learning_rate=1e-4,
    weight_decay=1e-4,
    step_size=5,
    gamma=0.5,
    optimizer_type='adam',
    print_model_summary=True
)

# Returns dictionary with all components:
# {
#   'model': model_on_device,
#   'device': device,
#   'criterion': CrossEntropyLoss,
#   'optimizer': Adam,
#   'scheduler': StepLR,
#   'summary': model_summary_dict
# }
```

## Complete Training Example

```python
from model_setup import setup_model_for_training
from model_custom_cnn import CropShieldCNN
from fast_dataset import make_loaders

# 1. Load data
train_loader, val_loader, test_loader, class_names, info = make_loaders(
    batch_size=32,
    num_workers=8,
    augmentation_mode='moderate'
)

# 2. Setup model
model = CropShieldCNN(num_classes=22)
setup = setup_model_for_training(model)

# Extract components
model = setup['model']
device = setup['device']
criterion = setup['criterion']
optimizer = setup['optimizer']
scheduler = setup['scheduler']

# 3. Training loop
num_epochs = 100
best_acc = 0.0

for epoch in range(num_epochs):
    # Train
    model.train()
    train_loss = 0.0
    train_correct = 0
    
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        
        # Forward
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        # Backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Metrics
        train_loss += loss.item()
        _, predicted = outputs.max(1)
        train_correct += predicted.eq(labels).sum().item()
    
    train_acc = 100.0 * train_correct / info['train_size']
    
    # Validate
    model.eval()
    val_correct = 0
    
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = outputs.max(1)
            val_correct += predicted.eq(labels).sum().item()
    
    val_acc = 100.0 * val_correct / info['val_size']
    
    # Update learning rate
    scheduler.step()
    
    # Save best model
    if val_acc > best_acc:
        best_acc = val_acc
        torch.save(model.state_dict(), 'best_cropshield_cnn.pth')
    
    print(f"Epoch {epoch+1}/{num_epochs} - "
          f"Train Acc: {train_acc:.2f}%, Val Acc: {val_acc:.2f}%")

print(f"Best Validation Accuracy: {best_acc:.2f}%")
```

## Memory Requirements

### Custom CNN (4.7M parameters)
- **Parameter Memory**: 17.92 MB
- **Activation Memory (batch=1)**: 155.04 MB
- **Activation Memory (batch=32)**: ~4.98 GB
- **Total GPU Memory (batch=32)**: ~5.0 GB
- **Fits RTX 4060**: ✅ Yes (8GB available)

### EfficientNet-B0 (4.0M parameters)
- **Parameter Memory**: ~15.4 MB
- **Activation Memory (batch=32)**: ~3.5 GB
- **Total GPU Memory (batch=32)**: ~3.5 GB
- **Fits RTX 4060**: ✅ Yes (8GB available)

## Learning Rate Schedule

With **StepLR** (step_size=5, gamma=0.5):

| Epoch Range | Learning Rate |
|-------------|---------------|
| 1-5         | 1e-4          |
| 6-10        | 5e-5          |
| 11-15       | 2.5e-5        |
| 16-20       | 1.25e-5       |
| 21-25       | 6.25e-6       |
| 26+         | 3.125e-6      |

## Optimizer Comparison

### Adam (Recommended)
- ✅ Adaptive learning rates per parameter
- ✅ Works well with default settings
- ✅ Good for both custom CNN and transfer learning
- ✅ Less sensitive to LR tuning

### SGD with Momentum
```python
setup = setup_model_for_training(
    model=model,
    learning_rate=1e-3,      # Higher LR for SGD
    optimizer_type='sgd'     # Uses momentum=0.9
)
```
- Requires careful LR tuning
- May converge faster with proper tuning
- More sensitive to hyperparameters

## Next Steps

1. **Test the setup**: Run `python model_setup.py` to verify all functions work
2. **Create training script**: Build `train_custom_cnn.py` with full training loop
3. **Monitor training**: Use TensorBoard or matplotlib for loss/accuracy curves
4. **Evaluate**: Test on validation set and generate confusion matrix
5. **Deploy**: Load best model checkpoint for inference

## Troubleshooting

### Out of Memory (OOM)
```python
# Reduce batch size
train_loader, val_loader, test_loader, _, _ = make_loaders(
    batch_size=16,  # Reduced from 32
    num_workers=8
)
```

### Slow Convergence
```python
# Try higher learning rate
setup = setup_model_for_training(model, learning_rate=1e-3)
```

### Model Not Learning
- Check data normalization (should use ImageNet stats)
- Verify augmentation isn't too aggressive
- Check loss is decreasing
- Verify labels are correct

## References

- **Custom CNN**: `model_custom_cnn.py` - 4.7M parameters
- **Data Loading**: `fast_dataset.py` - Optimized data pipeline
- **Augmentation**: `transforms.py` - MODERATE mode recommended
- **Architecture Analysis**: `CNN_ARCHITECTURE_ANALYSIS.md` - Detailed comparisons
