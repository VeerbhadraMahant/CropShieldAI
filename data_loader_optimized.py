"""
CropShield AI - Optimized PyTorch Data Loader
Based on profiling results, optimized for Windows multiprocessing
"""

import torch
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
import os

def create_optimized_dataloaders(
    data_dir='Database/',
    img_size=224,
    batch_size=32,
    train_split=0.8,
    val_split=0.1,
    test_split=0.1,
    num_workers=0,  # Set to 0 for Windows compatibility
    seed=42
):
    """
    Creates optimized train/val/test DataLoaders for CropShield AI
    
    Args:
        data_dir: Path to Database folder
        img_size: Target image size (default 224 for transfer learning)
        batch_size: Batch size for training
        train_split: Proportion for training set
        val_split: Proportion for validation set
        test_split: Proportion for test set
        num_workers: Number of worker processes (0 for Windows)
        seed: Random seed for reproducibility
    
    Returns:
        train_loader, val_loader, test_loader, class_names, dataset_info
    """
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🖥️  Device: {device}")
    if torch.cuda.is_available():
        print(f"   GPU: {torch.cuda.get_device_name(0)}")
    
    # Verify splits sum to 1.0
    assert abs(train_split + val_split + test_split - 1.0) < 1e-6, "Splits must sum to 1.0"
    
    # Data transforms
    # Training transforms with augmentation
    train_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  # ImageNet stats
    ])
    
    # Validation/Test transforms (no augmentation)
    eval_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    # Load full dataset with eval transform first
    print(f"\n📁 Loading dataset from: {data_dir}")
    full_dataset = datasets.ImageFolder(data_dir, transform=eval_transform)
    
    class_names = full_dataset.classes
    num_classes = len(class_names)
    total_images = len(full_dataset)
    
    print(f"✅ Total images: {total_images:,}")
    print(f"✅ Number of classes: {num_classes}")
    print(f"✅ Classes: {class_names}")
    
    # Calculate split sizes
    train_size = int(train_split * total_images)
    val_size = int(val_split * total_images)
    test_size = total_images - train_size - val_size  # Ensure all images are used
    
    print(f"\n📊 Dataset Split:")
    print(f"   Training:   {train_size:,} images ({train_split*100:.0f}%)")
    print(f"   Validation: {val_size:,} images ({val_split*100:.0f}%)")
    print(f"   Test:       {test_size:,} images ({test_split*100:.0f}%)")
    
    # Split dataset with reproducible seed
    torch.manual_seed(seed)
    train_dataset, val_dataset, test_dataset = random_split(
        full_dataset,
        [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(seed)
    )
    
    # Apply training transforms to training set only
    # This is a workaround since random_split doesn't support different transforms
    train_dataset.dataset = datasets.ImageFolder(data_dir, transform=train_transform)
    
    # Create DataLoaders with optimized settings
    print(f"\n⚙️  DataLoader Configuration:")
    print(f"   Batch size: {batch_size}")
    print(f"   num_workers: {num_workers}")
    print(f"   pin_memory: {device.type == 'cuda'}")
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,  # Shuffle training data
        num_workers=num_workers,
        pin_memory=(device.type == 'cuda'),
        persistent_workers=False  # Must be False when num_workers=0
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,  # Don't shuffle validation
        num_workers=num_workers,
        pin_memory=(device.type == 'cuda'),
        persistent_workers=False
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,  # Don't shuffle test
        num_workers=num_workers,
        pin_memory=(device.type == 'cuda'),
        persistent_workers=False
    )
    
    # Dataset info dictionary
    dataset_info = {
        'num_classes': num_classes,
        'class_names': class_names,
        'total_images': total_images,
        'train_size': train_size,
        'val_size': val_size,
        'test_size': test_size,
        'img_size': img_size,
        'device': device
    }
    
    print(f"\n✅ DataLoaders created successfully!")
    print(f"   Train batches: {len(train_loader)}")
    print(f"   Val batches:   {len(val_loader)}")
    print(f"   Test batches:  {len(test_loader)}")
    
    return train_loader, val_loader, test_loader, class_names, dataset_info


if __name__ == '__main__':
    """
    Test the optimized dataloaders
    """
    import time
    
    print("="*80)
    print("CROPSHIELD AI - OPTIMIZED DATA LOADER TEST")
    print("="*80)
    
    # Create dataloaders
    train_loader, val_loader, test_loader, class_names, info = create_optimized_dataloaders(
        data_dir='Database/',
        img_size=224,
        batch_size=32,
        train_split=0.8,
        val_split=0.1,
        test_split=0.1,
        num_workers=0,  # Set to 0 for Windows
        seed=42
    )
    
    print("\n" + "="*80)
    print("TESTING DATA LOADING PERFORMANCE")
    print("="*80)
    
    # Test loading speed
    print("\n🔄 Loading first batch from each split...")
    
    # Train
    start = time.time()
    images, labels = next(iter(train_loader))
    train_time = time.time() - start
    print(f"\n✅ Training batch:")
    print(f"   Shape: {images.shape}")
    print(f"   Labels: {labels.shape}")
    print(f"   Load time: {train_time*1000:.1f}ms")
    print(f"   Device: {images.device}")
    
    # Validation
    start = time.time()
    images, labels = next(iter(val_loader))
    val_time = time.time() - start
    print(f"\n✅ Validation batch:")
    print(f"   Shape: {images.shape}")
    print(f"   Labels: {labels.shape}")
    print(f"   Load time: {val_time*1000:.1f}ms")
    
    # Test
    start = time.time()
    images, labels = next(iter(test_loader))
    test_time = time.time() - start
    print(f"\n✅ Test batch:")
    print(f"   Shape: {images.shape}")
    print(f"   Labels: {labels.shape}")
    print(f"   Load time: {test_time*1000:.1f}ms")
    
    # Class distribution check
    print("\n" + "="*80)
    print("CLASS DISTRIBUTION CHECK")
    print("="*80)
    
    print("\n📊 Sampling 100 batches from training set...")
    class_counts = torch.zeros(info['num_classes'])
    
    for i, (_, labels) in enumerate(train_loader):
        if i >= 100:
            break
        for label in labels:
            class_counts[label] += 1
    
    print("\nClass distribution in sample:")
    for idx, (class_name, count) in enumerate(zip(class_names[:10], class_counts[:10])):
        print(f"   {class_name}: {int(count)} samples")
    
    if len(class_names) > 10:
        print(f"   ... and {len(class_names) - 10} more classes")
    
    print("\n" + "="*80)
    print("✅ DATA LOADER TEST COMPLETE!")
    print("="*80)
    
    print("\n💡 USAGE IN TRAINING SCRIPT:")
    print("""
from data_loader_optimized import create_optimized_dataloaders

# Create dataloaders
train_loader, val_loader, test_loader, class_names, info = create_optimized_dataloaders(
    data_dir='Database/',
    img_size=224,
    batch_size=64,  # Increase if GPU memory allows
    num_workers=0   # Keep at 0 for Windows
)

# Training loop
for epoch in range(num_epochs):
    for images, labels in train_loader:
        images = images.to(device)
        labels = labels.to(device)
        # Your training code here...
    """)
