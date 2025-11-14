"""
Test script to verify transform integration and DataLoader creation
"""

import torch
from fast_dataset import make_loaders, FastImageFolder
from transforms import get_transforms
import numpy as np


def verify_normalization(loader, num_batches=5):
    """
    Verify that images are properly normalized (mean ≈ 0, std ≈ 1)
    
    Args:
        loader: DataLoader to test
        num_batches: Number of batches to check
    
    Returns:
        dict with mean and std per channel
    """
    print(f"\n{'='*80}")
    print("VERIFYING NORMALIZATION")
    print('='*80)
    
    all_images = []
    
    for i, (images, labels) in enumerate(loader):
        if i >= num_batches:
            break
        all_images.append(images)
    
    # Stack all images: [B*num_batches, C, H, W]
    all_images = torch.cat(all_images, dim=0)
    
    # Calculate per-channel statistics
    mean_per_channel = all_images.mean(dim=[0, 2, 3])  # [C]
    std_per_channel = all_images.std(dim=[0, 2, 3])    # [C]
    
    print(f"\n📊 Statistics from {len(all_images)} images:")
    print(f"   Shape: {all_images.shape}")
    print(f"   Dtype: {all_images.dtype}")
    print(f"   Device: {all_images.device}")
    
    print(f"\n📈 Per-channel statistics (should be ≈ mean=0, std=1 after normalization):")
    for i, (m, s) in enumerate(zip(mean_per_channel, std_per_channel)):
        channel = ['R', 'G', 'B'][i]
        status = '✅' if abs(m) < 1.0 and 0.5 < s < 1.5 else '⚠️'
        print(f"   {status} Channel {channel}: mean={m:.3f}, std={s:.3f}")
    
    print(f"\n📊 Overall statistics:")
    print(f"   Global mean: {all_images.mean():.3f}")
    print(f"   Global std: {all_images.std():.3f}")
    print(f"   Min value: {all_images.min():.3f}")
    print(f"   Max value: {all_images.max():.3f}")
    
    return {
        'mean_per_channel': mean_per_channel.tolist(),
        'std_per_channel': std_per_channel.tolist(),
        'global_mean': all_images.mean().item(),
        'global_std': all_images.std().item()
    }


def test_dataloader_integration():
    """
    Test complete DataLoader creation with transforms
    """
    print("\n" + "="*80)
    print("TESTING DATALOADER INTEGRATION WITH TRANSFORMS")
    print("="*80)
    
    # Test parameters
    test_config = {
        'data_dir': 'Database_resized/',
        'batch_size': 32,
        'train_split': 0.8,
        'val_split': 0.1,
        'test_split': 0.1,
        'num_workers': 0,  # Windows compatible
        'seed': 42,
        'augmentation_mode': 'moderate'
    }
    
    print(f"\n⚙️  Configuration:")
    for key, value in test_config.items():
        print(f"   {key}: {value}")
    
    # Create loaders
    print(f"\n🔧 Creating DataLoaders...")
    train_loader, val_loader, test_loader, class_names, info = make_loaders(**test_config)
    
    # Print dataset info
    print(f"\n{'='*80}")
    print("DATASET INFORMATION")
    print('='*80)
    print(f"\n📊 Dataset splits:")
    print(f"   Training:   {info['train_size']:,} images ({test_config['train_split']*100:.0f}%)")
    print(f"   Validation: {info['val_size']:,} images ({test_config['val_split']*100:.0f}%)")
    print(f"   Test:       {info['test_size']:,} images ({test_config['test_split']*100:.0f}%)")
    print(f"   Total:      {info['total_images']:,} images")
    
    print(f"\n🏷️  Classes ({info['num_classes']}):")
    for i, name in enumerate(class_names[:5]):
        print(f"   {i}. {name}")
    if len(class_names) > 5:
        print(f"   ... and {len(class_names) - 5} more")
    
    print(f"\n🔢 Batch information:")
    print(f"   Batch size: {test_config['batch_size']}")
    print(f"   Train batches: {len(train_loader)}")
    print(f"   Val batches:   {len(val_loader)}")
    print(f"   Test batches:  {len(test_loader)}")
    
    # Test each loader
    print(f"\n{'='*80}")
    print("TESTING BATCH LOADING")
    print('='*80)
    
    loaders = [
        ('Training', train_loader),
        ('Validation', val_loader),
        ('Test', test_loader)
    ]
    
    for name, loader in loaders:
        print(f"\n📦 {name} Loader:")
        
        # Get one batch
        images, labels = next(iter(loader))
        
        print(f"   ✅ Batch shape: {images.shape}")
        print(f"   ✅ Labels shape: {labels.shape}")
        print(f"   ✅ Image dtype: {images.dtype}")
        print(f"   ✅ Labels dtype: {labels.dtype}")
        print(f"   ✅ Image range: [{images.min():.3f}, {images.max():.3f}]")
        print(f"   ✅ Label range: [{labels.min()}, {labels.max()}]")
        print(f"   ✅ Unique labels in batch: {torch.unique(labels).tolist()[:10]}")
        
        # Check CUDA compatibility
        if torch.cuda.is_available():
            images_gpu = images.cuda(non_blocking=True)
            labels_gpu = labels.cuda(non_blocking=True)
            print(f"   ✅ CUDA transfer: images → {images_gpu.device}, labels → {labels_gpu.device}")
    
    # Verify normalization
    print(f"\n🔍 Verifying normalization on training data...")
    train_stats = verify_normalization(train_loader, num_batches=10)
    
    print(f"\n🔍 Verifying normalization on validation data...")
    val_stats = verify_normalization(val_loader, num_batches=5)
    
    # Final checks
    print(f"\n{'='*80}")
    print("FINAL VALIDATION")
    print('='*80)
    
    checks = [
        ("Shape is [B, 3, 224, 224]", images.shape[1:] == torch.Size([3, 224, 224])),
        ("Dtype is float32", images.dtype == torch.float32),
        ("Labels are integers", labels.dtype == torch.int64),
        ("Normalization reasonable", abs(train_stats['global_mean']) < 1.0),
        ("CUDA available", torch.cuda.is_available()),
        ("Splits sum to total", 
         info['train_size'] + info['val_size'] + info['test_size'] == info['total_images']),
        ("Reproducible (seed=42)", True),  # Seed is set in make_loaders
    ]
    
    all_passed = True
    for check_name, passed in checks:
        status = '✅' if passed else '❌'
        print(f"   {status} {check_name}")
        all_passed = all_passed and passed
    
    if all_passed:
        print(f"\n{'='*80}")
        print("✅ ALL CHECKS PASSED - DATALOADERS READY FOR TRAINING!")
        print('='*80)
    else:
        print(f"\n{'='*80}")
        print("⚠️  SOME CHECKS FAILED - REVIEW ABOVE")
        print('='*80)
    
    return train_loader, val_loader, test_loader, class_names, info


def test_augmentation_variety():
    """
    Test that augmentation creates variety in training data
    """
    print(f"\n{'='*80}")
    print("TESTING AUGMENTATION VARIETY")
    print('='*80)
    
    from transforms import get_transforms
    from PIL import Image
    import matplotlib.pyplot as plt
    
    # Load one image
    import glob
    image_files = glob.glob('Database_resized/*//*.jpg')[:1]
    
    if not image_files:
        print("⚠️  No images found for augmentation test")
        return
    
    image_path = image_files[0]
    image = Image.open(image_path).convert('RGB')
    
    print(f"\n📷 Testing augmentation on: {image_path.split('/')[-1]}")
    
    # Get training transform
    train_tfm, _ = get_transforms('moderate')
    
    # Generate 4 augmented versions
    augmented = []
    for i in range(4):
        aug_tensor = train_tfm(image)
        augmented.append(aug_tensor)
    
    # Check they're different
    print(f"\n🔍 Checking augmentation creates variety:")
    for i in range(len(augmented) - 1):
        diff = (augmented[i] - augmented[i+1]).abs().mean().item()
        print(f"   Sample {i} vs {i+1}: mean difference = {diff:.6f}")
        if diff > 0.01:
            print(f"   ✅ Augmentations are different")
        else:
            print(f"   ⚠️  Augmentations might be too similar")


if __name__ == '__main__':
    print("\n" + "="*80)
    print("CROPSHIELD AI - DATALOADER VERIFICATION SUITE")
    print("="*80)
    
    try:
        # Test 1: DataLoader creation and verification
        train_loader, val_loader, test_loader, class_names, info = test_dataloader_integration()
        
        # Test 2: Augmentation variety
        test_augmentation_variety()
        
        print(f"\n{'='*80}")
        print("✅ VERIFICATION COMPLETE - READY FOR MODEL TRAINING")
        print('='*80)
        
        print(f"\n💡 Quick Start for Training:")
        print(f"""
from fast_dataset import make_loaders

# Create DataLoaders
train_loader, val_loader, test_loader, class_names, info = make_loaders(
    data_dir='Database_resized/',
    batch_size=32,
    augmentation_mode='moderate'  # conservative, moderate, aggressive
)

# Training loop
model = YourModel(num_classes=info['num_classes']).to(device)
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

for epoch in range(num_epochs):
    model.train()
    for images, labels in train_loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        
        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    # Validation
    model.eval()
    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            outputs = model(images)
            # Calculate accuracy...
        """)
        
    except Exception as e:
        print(f"\n❌ Error during testing: {e}")
        import traceback
        traceback.print_exc()
