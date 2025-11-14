"""
Simple Example: Using Model Factory
====================================

Demonstrates how to use the model factory for Phase 4 training.
"""

import torch
from models.model_factory import get_model, print_model_comparison

print("\n" + "="*60)
print("🎯 MODEL FACTORY USAGE EXAMPLE")
print("="*60)

# Show available models
print_model_comparison()

# Example 1: Create Custom CNN
print("\n" + "="*60)
print("EXAMPLE 1: Custom CNN")
print("="*60)

model, optimizer, criterion, scheduler, device = get_model(
    model_type='custom',
    learning_rate=1e-4,
    verbose=True
)

print("\n✅ Custom CNN ready!")
print("   Use this model for baseline training (75-85% accuracy)")

# Test forward pass
print("\n📝 Testing forward pass...")
x = torch.randn(8, 3, 224, 224).to(device)
model.eval()
with torch.no_grad():
    output = model(x)
    probs = torch.softmax(output, dim=1)
    pred_classes = torch.argmax(probs, dim=1)

print(f"✅ Input shape:       {list(x.shape)}")
print(f"✅ Output shape:      {list(output.shape)}")
print(f"✅ Probabilities:     {list(probs.shape)}")
print(f"✅ Predicted classes: {pred_classes.tolist()}")

# Example 2: Create EfficientNet-B0
print("\n" + "="*60)
print("EXAMPLE 2: EfficientNet-B0 (Transfer Learning)")
print("="*60)

model_eff, optimizer_eff, criterion_eff, scheduler_eff, device = get_model(
    model_type='efficientnet',
    learning_rate=1e-4,
    pretrained=True,
    verbose=True
)

print("\n✅ EfficientNet-B0 ready!")
print("   Use this model for state-of-the-art performance (92-96% accuracy)")

# Test forward pass
print("\n📝 Testing forward pass...")
model_eff.eval()
with torch.no_grad():
    output_eff = model_eff(x)
    probs_eff = torch.softmax(output_eff, dim=1)
    pred_classes_eff = torch.argmax(probs_eff, dim=1)

print(f"✅ Input shape:       {list(x.shape)}")
print(f"✅ Output shape:      {list(output_eff.shape)}")
print(f"✅ Probabilities:     {list(probs_eff.shape)}")
print(f"✅ Predicted classes: {pred_classes_eff.tolist()}")

# Example 3: Minimal usage for training
print("\n" + "="*60)
print("EXAMPLE 3: Minimal Usage for Training Loop")
print("="*60)

print("\n📝 Code snippet:")
print("""
from models import get_model
from fast_dataset import make_loaders

# Create model (one line!)
model, optimizer, criterion, scheduler, device = get_model('custom')

# Load data
train_loader, val_loader, test_loader, class_names, info = make_loaders(
    batch_size=32,
    num_workers=8,
    augmentation_mode='moderate'
)

# Training loop
for epoch in range(100):
    model.train()
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    scheduler.step()
""")

# Summary
print("\n" + "="*60)
print("📋 SUMMARY")
print("="*60)
print("\n✅ Model factory provides:")
print("   • Unified interface for different architectures")
print("   • Automatic device detection (GPU/CPU)")
print("   • Complete training setup (optimizer, loss, scheduler)")
print("   • Ready for Phase 4 training")
print("\n💡 Two models available:")
print("   1. Custom CNN:      75-85% accuracy, train from scratch")
print("   2. EfficientNet-B0: 92-96% accuracy, transfer learning")
print("\n🚀 Next step: Create training script for Phase 4!")
print("="*60)
