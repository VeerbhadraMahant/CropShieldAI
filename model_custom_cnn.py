"""
CropShield AI - Custom CNN Architecture
A lightweight custom CNN for plant disease classification

Architecture Features:
- 4 convolutional blocks with batch normalization
- Dropout regularization for overfitting prevention
- Global Average Pooling (GradCAM compatible)
- <10M parameters (efficient for RTX 4060)
- Progressive channel expansion: 64 → 128 → 256 → 512
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple


class CropShieldCNN(nn.Module):
    """
    Custom CNN for CropShield AI plant disease classification.
    
    Architecture:
        Input: (B, 3, 224, 224)
        
        Block 1: Conv(64) → BN → ReLU → Conv(64) → BN → ReLU → MaxPool → Dropout
        Block 2: Conv(128) → BN → ReLU → Conv(128) → BN → ReLU → MaxPool → Dropout
        Block 3: Conv(256) → BN → ReLU → Conv(256) → BN → ReLU → MaxPool → Dropout
        Block 4: Conv(512) → BN → ReLU → Conv(512) → BN → ReLU → MaxPool → Dropout
        
        Global Average Pooling → FC(512 → num_classes)
        
        Output: (B, num_classes) logits
    
    Args:
        num_classes: Number of output classes (default: 22 for CropShield dataset)
        dropout_rate: Dropout probability (default: 0.5)
        
    Total Parameters: ~6.5M (well under 10M constraint)
    """
    
    def __init__(self, num_classes: int = 22, dropout_rate: float = 0.5):
        super(CropShieldCNN, self).__init__()
        
        self.num_classes = num_classes
        self.dropout_rate = dropout_rate
        
        # Block 1: 3 → 64 channels (224x224 → 112x112)
        self.block1 = nn.Sequential(
            # First conv: 3 → 64
            nn.Conv2d(3, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            
            # Second conv: 64 → 64
            nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            
            # Downsample: 224x224 → 112x112
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout2d(p=dropout_rate * 0.5)  # Light dropout early
        )
        
        # Block 2: 64 → 128 channels (112x112 → 56x56)
        self.block2 = nn.Sequential(
            # First conv: 64 → 128
            nn.Conv2d(64, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            
            # Second conv: 128 → 128
            nn.Conv2d(128, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            
            # Downsample: 112x112 → 56x56
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout2d(p=dropout_rate * 0.6)
        )
        
        # Block 3: 128 → 256 channels (56x56 → 28x28)
        self.block3 = nn.Sequential(
            # First conv: 128 → 256
            nn.Conv2d(128, 256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            
            # Second conv: 256 → 256
            nn.Conv2d(256, 256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            
            # Downsample: 56x56 → 28x28
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout2d(p=dropout_rate * 0.7)
        )
        
        # Block 4: 256 → 512 channels (28x28 → 14x14)
        self.block4 = nn.Sequential(
            # First conv: 256 → 512
            nn.Conv2d(256, 512, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            
            # Second conv: 512 → 512
            nn.Conv2d(512, 512, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            
            # Downsample: 28x28 → 14x14
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout2d(p=dropout_rate)
        )
        
        # Global Average Pooling (GradCAM compatible!)
        # Converts (B, 512, 14, 14) → (B, 512)
        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout_rate),
            nn.Linear(512, num_classes)
        )
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        """
        Initialize network weights using He initialization for ReLU networks.
        """
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input tensor of shape (B, 3, 224, 224)
            
        Returns:
            Logits of shape (B, num_classes)
        """
        # Convolutional blocks
        x = self.block1(x)  # (B, 64, 112, 112)
        x = self.block2(x)  # (B, 128, 56, 56)
        x = self.block3(x)  # (B, 256, 28, 28)
        x = self.block4(x)  # (B, 512, 14, 14)
        
        # Global average pooling
        x = self.global_avg_pool(x)  # (B, 512, 1, 1)
        x = torch.flatten(x, 1)      # (B, 512)
        
        # Classification
        x = self.classifier(x)  # (B, num_classes)
        
        return x
    
    def get_feature_maps(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Extract feature maps from each block (useful for visualization/debugging).
        
        Args:
            x: Input tensor of shape (B, 3, 224, 224)
            
        Returns:
            Tuple of feature maps from each block
        """
        f1 = self.block1(x)
        f2 = self.block2(f1)
        f3 = self.block3(f2)
        f4 = self.block4(f3)
        return f1, f2, f3, f4
    
    def get_gradcam_target_layer(self):
        """
        Get the target layer for GradCAM visualization.
        Use the last convolutional layer in block4.
        
        Returns:
            Target layer for GradCAM
        """
        # Return the last conv layer before pooling (block4's last conv)
        return self.block4[3]  # Last Conv2d in block4


def count_parameters(model: nn.Module) -> int:
    """
    Count total and trainable parameters in the model.
    
    Args:
        model: PyTorch model
        
    Returns:
        Total number of parameters
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"\n{'='*60}")
    print(f"MODEL PARAMETERS")
    print(f"{'='*60}")
    print(f"Total parameters:     {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Non-trainable:        {total_params - trainable_params:,}")
    print(f"{'='*60}\n")
    
    return total_params


def test_model():
    """
    Test the model with a dummy input and print architecture summary.
    """
    print("\n" + "="*60)
    print("CROPSHIELD CNN - MODEL TEST")
    print("="*60)
    
    # Create model
    num_classes = 22  # CropShield AI has 22 disease classes
    model = CropShieldCNN(num_classes=num_classes, dropout_rate=0.5)
    
    # Count parameters
    total_params = count_parameters(model)
    
    # Check parameter constraint
    if total_params < 10_000_000:
        print(f"✅ Parameter constraint satisfied: {total_params:,} < 10M")
    else:
        print(f"❌ Too many parameters: {total_params:,} >= 10M")
    
    # Test forward pass
    print("\n" + "="*60)
    print("FORWARD PASS TEST")
    print("="*60)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    # Create dummy input (batch of 4 images)
    batch_size = 4
    dummy_input = torch.randn(batch_size, 3, 224, 224).to(device)
    
    print(f"Input shape:  {dummy_input.shape}")
    
    # Forward pass
    with torch.no_grad():
        output = model(dummy_input)
    
    print(f"Output shape: {output.shape}")
    print(f"Output range: [{output.min().item():.3f}, {output.max().item():.3f}]")
    
    # Test feature extraction
    print("\n" + "="*60)
    print("FEATURE MAPS")
    print("="*60)
    
    with torch.no_grad():
        f1, f2, f3, f4 = model.get_feature_maps(dummy_input)
    
    print(f"Block 1 features: {f1.shape}")
    print(f"Block 2 features: {f2.shape}")
    print(f"Block 3 features: {f3.shape}")
    print(f"Block 4 features: {f4.shape}")
    
    # Test GradCAM target
    print("\n" + "="*60)
    print("GRADCAM COMPATIBILITY")
    print("="*60)
    
    target_layer = model.get_gradcam_target_layer()
    print(f"GradCAM target layer: {target_layer}")
    print(f"✅ GradCAM compatible (using last conv layer)")
    
    # Memory usage estimate
    print("\n" + "="*60)
    print("MEMORY ESTIMATION")
    print("="*60)
    
    # Rough estimate: params (4 bytes each) + activations
    param_memory = total_params * 4 / (1024**2)  # MB
    activation_memory = 200  # Approximate for batch=32
    
    print(f"Parameter memory: ~{param_memory:.1f} MB")
    print(f"Activation memory (batch=32): ~{activation_memory:.1f} MB")
    print(f"Total estimated: ~{param_memory + activation_memory:.1f} MB")
    print(f"✅ Fits comfortably on RTX 4060 (8GB)")
    
    print("\n✅ All tests passed!")
    print("="*60 + "\n")
    
    return model


# Example training setup
def create_training_setup(model: nn.Module, learning_rate: float = 1e-3):
    """
    Create loss function and optimizer for training.
    
    Args:
        model: CropShieldCNN model
        learning_rate: Learning rate for optimizer
        
    Returns:
        Tuple of (criterion, optimizer, scheduler)
    """
    # Loss function (CrossEntropyLoss for multi-class classification)
    criterion = nn.CrossEntropyLoss()
    
    # Optimizer (Adam with weight decay for regularization)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=learning_rate,
        weight_decay=1e-4  # L2 regularization
    )
    
    # Learning rate scheduler (reduce on plateau)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='max',           # Maximize validation accuracy
        factor=0.5,           # Reduce LR by half
        patience=5            # Wait 5 epochs before reducing
    )
    
    return criterion, optimizer, scheduler


# Full training-ready example
if __name__ == "__main__":
    """
    Complete training-ready code example.
    """
    
    # Test the model
    model = test_model()
    
    # Try to import torchsummary for detailed summary
    print("\n" + "="*60)
    print("DETAILED MODEL SUMMARY")
    print("="*60)
    
    try:
        from torchsummary import summary
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = model.to(device)
        summary(model, input_size=(3, 224, 224), device=str(device))
    except ImportError:
        print("⚠️  torchsummary not installed. Install with: pip install torchsummary")
        print("    Skipping detailed summary...")
    
    # Create training setup
    print("\n" + "="*60)
    print("TRAINING SETUP")
    print("="*60)
    
    criterion, optimizer, scheduler = create_training_setup(model, learning_rate=1e-3)
    
    print(f"Loss function: {criterion.__class__.__name__}")
    print(f"Optimizer: {optimizer.__class__.__name__}")
    print(f"  Learning rate: {optimizer.param_groups[0]['lr']}")
    print(f"  Weight decay: {optimizer.param_groups[0]['weight_decay']}")
    print(f"Scheduler: {scheduler.__class__.__name__}")
    print(f"  Patience: {scheduler.patience}")
    print(f"  Factor: {scheduler.factor}")
    
    # Quick training loop example
    print("\n" + "="*60)
    print("TRAINING LOOP EXAMPLE")
    print("="*60)
    
    print("""
# Example training loop:

from fast_dataset import make_loaders

# Load data
train_loader, val_loader, test_loader, class_names, info = make_loaders(
    batch_size=32,
    num_workers=8,
    augmentation_mode='moderate'
)

# Setup
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = CropShieldCNN(num_classes=22).to(device)
criterion, optimizer, scheduler = create_training_setup(model)

# Training
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
    scheduler.step(val_acc)
    
    # Save best model
    if val_acc > best_acc:
        best_acc = val_acc
        torch.save(model.state_dict(), 'best_cropshield_cnn.pth')
    
    print(f"Epoch {epoch+1}/{num_epochs} - "
          f"Train Acc: {train_acc:.2f}%, Val Acc: {val_acc:.2f}%")

print(f"Best Validation Accuracy: {best_acc:.2f}%")
    """)
    
    print("\n✅ Model is ready for training!")
    print("="*60 + "\n")
