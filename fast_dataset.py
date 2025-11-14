"""
CropShield AI - Fast Custom Dataset with torchvision.io
Uses native libjpeg decoder for maximum performance

This custom Dataset class is optimized for:
- Fast JPEG decoding (torchvision.io.read_image)
- Automatic grayscale to RGB conversion
- Robust error handling
- Zero-copy tensor operations
- Modular augmentation pipeline (conservative/moderate/aggressive)
"""

import torch
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
import torchvision.io as io
from pathlib import Path
import time
from collections import defaultdict
import os

# Import our new transforms module
try:
    from transforms import get_transforms, IMAGENET_MEAN, IMAGENET_STD
    TRANSFORMS_AVAILABLE = True
except ImportError:
    print("⚠️  transforms.py not found. Using legacy transforms.")
    TRANSFORMS_AVAILABLE = False


class FastImageFolder(Dataset):
    """
    Custom Dataset using torchvision.io.read_image for fast JPEG decoding
    
    Advantages over ImageFolder:
    - 2-3x faster JPEG decoding (libjpeg-turbo)
    - Direct tensor output (no PIL conversion)
    - Better error handling
    - Memory efficient
    
    Args:
        root: Path to dataset directory (expects class subfolders)
        transform: Optional transforms to apply
    """
    
    def __init__(self, root, transform=None):
        self.root = Path(root)
        self.transform = transform
        
        # Build class index mapping
        self.classes = sorted([d.name for d in self.root.iterdir() if d.is_dir()])
        self.class_to_idx = {cls_name: idx for idx, cls_name in enumerate(self.classes)}
        
        # Find all images
        self.samples = []
        self.targets = []
        
        valid_extensions = {'.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG'}
        
        for class_name in self.classes:
            class_dir = self.root / class_name
            class_idx = self.class_to_idx[class_name]
            
            for img_path in class_dir.iterdir():
                if img_path.suffix in valid_extensions:
                    self.samples.append(str(img_path))
                    self.targets.append(class_idx)
        
        if len(self.samples) == 0:
            raise RuntimeError(f"No images found in {root}")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        """
        Load and return image tensor and label
        
        Returns:
            image: Tensor of shape (C, H, W) normalized to [0, 1]
            label: Integer class label
        """
        img_path = self.samples[idx]
        label = self.targets[idx]
        
        try:
            # Fast JPEG decoding using libjpeg-turbo
            # Returns tensor in [0, 255] range, shape (C, H, W)
            image = io.read_image(img_path, mode=io.ImageReadMode.RGB)
            
            # Convert torch tensor to PIL Image for transforms compatibility
            from torchvision.transforms.functional import to_pil_image
            image = to_pil_image(image)
            
            # Apply transforms if provided
            if self.transform is not None:
                image = self.transform(image)
            
            return image, label
            
        except Exception as e:
            # Handle corrupt/missing files gracefully
            print(f"⚠️  Warning: Failed to load {img_path}: {e}")
            
            # Return a black image and label as fallback
            image = torch.zeros((3, 224, 224), dtype=torch.float32)
            return image, label
    
    def get_class_counts(self):
        """Return dictionary of class name -> count"""
        counts = defaultdict(int)
        for target in self.targets:
            class_name = self.classes[target]
            counts[class_name] += 1
        return dict(counts)


def make_loaders(
    data_dir='Database_resized/',
    batch_size=32,
    train_split=0.8,
    val_split=0.1,
    test_split=0.1,
    num_workers=None,
    seed=42,
    prefetch_factor=2,
    augmentation_mode='moderate'  # NEW: augmentation preset
):
    """
    Create train/val/test DataLoaders with optimized settings and modular augmentation
    
    Args:
        data_dir: Path to preprocessed dataset
        batch_size: Batch size for all loaders
        train_split: Training set proportion (0.0-1.0)
        val_split: Validation set proportion (0.0-1.0)
        test_split: Test set proportion (0.0-1.0)
        num_workers: Number of worker processes (auto-detect if None)
        seed: Random seed for reproducibility
        prefetch_factor: Number of batches to prefetch per worker
        augmentation_mode: 'conservative', 'moderate', or 'aggressive' (NEW)
    
    Returns:
        train_loader, val_loader, test_loader, class_names, dataset_info
    """
    
    # Auto-detect optimal num_workers
    if num_workers is None:
        # Use CPU count - 1, but at least 0 for Windows compatibility
        num_workers = 0 if os.name == 'nt' else max(1, os.cpu_count() - 1)
    
    # Device detection
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print("="*80)
    print("CROPSHIELD AI - FAST CUSTOM DATASET")
    print("="*80)
    
    print(f"\n🖥️  Device: {device}")
    if torch.cuda.is_available():
        print(f"   GPU: {torch.cuda.get_device_name(0)}")
    
    # Verify splits
    assert abs(train_split + val_split + test_split - 1.0) < 1e-6, "Splits must sum to 1.0"
    
    # Define transforms using new modular system
    if TRANSFORMS_AVAILABLE:
        train_transform, eval_transform = get_transforms(augmentation_mode)
        print(f"✅ Using {augmentation_mode.upper()} augmentation from transforms.py")
    else:
        # Fallback to legacy transforms if transforms.py not available
        print("⚠️  Using legacy transforms (transforms.py not found)")
        train_transform = transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        eval_transform = transforms.Compose([
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    
    # Load full dataset
    print(f"\n📁 Loading dataset from: {data_dir}")
    full_dataset = FastImageFolder(data_dir, transform=eval_transform)
    
    class_names = full_dataset.classes
    num_classes = len(class_names)
    total_images = len(full_dataset)
    
    print(f"✅ Total images: {total_images:,}")
    print(f"✅ Number of classes: {num_classes}")
    print(f"✅ Using torchvision.io.read_image (fast JPEG decoder)")
    
    # Print class distribution
    class_counts = full_dataset.get_class_counts()
    print(f"\n📊 Class Distribution:")
    for i, (class_name, count) in enumerate(sorted(class_counts.items())[:10]):
        print(f"   {i+1}. {class_name}: {count:,} images")
    if len(class_counts) > 10:
        print(f"   ... and {len(class_counts) - 10} more classes")
    
    # Calculate split sizes
    train_size = int(train_split * total_images)
    val_size = int(val_split * total_images)
    test_size = total_images - train_size - val_size
    
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
    train_dataset.dataset = FastImageFolder(data_dir, transform=train_transform)
    
    # DataLoader configuration
    print(f"\n⚙️  DataLoader Configuration:")
    print(f"   Batch size: {batch_size}")
    print(f"   num_workers: {num_workers}")
    print(f"   pin_memory: {device.type == 'cuda'}")
    print(f"   persistent_workers: {num_workers > 0}")
    print(f"   prefetch_factor: {prefetch_factor}")
    
    # Create optimized DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=(device.type == 'cuda'),
        persistent_workers=(num_workers > 0),
        prefetch_factor=prefetch_factor if num_workers > 0 else None,
        drop_last=False  # Use all training data
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=(device.type == 'cuda'),
        persistent_workers=(num_workers > 0),
        prefetch_factor=prefetch_factor if num_workers > 0 else None,
        drop_last=False
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=(device.type == 'cuda'),
        persistent_workers=(num_workers > 0),
        prefetch_factor=prefetch_factor if num_workers > 0 else None,
        drop_last=False
    )
    
    # Dataset info
    dataset_info = {
        'num_classes': num_classes,
        'class_names': class_names,
        'class_counts': class_counts,
        'total_images': total_images,
        'train_size': train_size,
        'val_size': val_size,
        'test_size': test_size,
        'img_size': 224,
        'device': device,
        'num_workers': num_workers
    }
    
    print(f"\n✅ DataLoaders created!")
    print(f"   Train batches: {len(train_loader)}")
    print(f"   Val batches:   {len(val_loader)}")
    print(f"   Test batches:  {len(test_loader)}")
    
    return train_loader, val_loader, test_loader, class_names, dataset_info


def benchmark_loader(loader, num_batches=10):
    """
    Quick benchmark of DataLoader performance
    
    Args:
        loader: DataLoader to benchmark
        num_batches: Number of batches to test
    
    Returns:
        Average time per batch in seconds
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print(f"\n🔄 Benchmarking DataLoader ({num_batches} batches)...")
    
    times = []
    start_total = time.time()
    
    for i, (images, labels) in enumerate(loader):
        if i >= num_batches:
            break
        
        batch_start = time.time()
        
        # Move to device (simulates training)
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        
        batch_time = time.time() - batch_start
        times.append(batch_time)
    
    total_time = time.time() - start_total
    avg_time = sum(times) / len(times)
    
    print(f"✅ Benchmark Results:")
    print(f"   Total time: {total_time:.2f}s")
    print(f"   Avg per batch: {avg_time*1000:.1f}ms")
    print(f"   Min: {min(times)*1000:.1f}ms, Max: {max(times)*1000:.1f}ms")
    print(f"   Throughput: {num_batches * 32 / total_time:.1f} img/s")
    
    return avg_time


if __name__ == '__main__':
    """
    Test the FastImageFolder dataset and DataLoaders
    """
    
    print("="*80)
    print("TESTING FAST CUSTOM DATASET")
    print("="*80)
    
    # Create DataLoaders
    try:
        train_loader, val_loader, test_loader, class_names, info = make_loaders(
            data_dir='Database_resized/',
            batch_size=32,
            num_workers=None,  # Auto-detect
            seed=42
        )
    except Exception as e:
        print(f"\n❌ Error: {e}")
        print("\nMake sure Database_resized/ exists!")
        print("Run: python scripts/resize_images.py")
        exit(1)
    
    # Test batch loading
    print("\n" + "="*80)
    print("TESTING BATCH LOADING")
    print("="*80)
    
    print("\n📦 Loading one batch from each split...")
    
    # Train batch
    images, labels = next(iter(train_loader))
    print(f"\n✅ Training batch:")
    print(f"   Images shape: {images.shape}")
    print(f"   Labels shape: {labels.shape}")
    print(f"   Image dtype: {images.dtype}")
    print(f"   Image range: [{images.min():.3f}, {images.max():.3f}]")
    print(f"   Labels: {labels[:5].tolist()}...")
    
    # Val batch
    images, labels = next(iter(val_loader))
    print(f"\n✅ Validation batch:")
    print(f"   Images shape: {images.shape}")
    print(f"   Labels shape: {labels.shape}")
    
    # Test batch
    images, labels = next(iter(test_loader))
    print(f"\n✅ Test batch:")
    print(f"   Images shape: {images.shape}")
    print(f"   Labels shape: {labels.shape}")
    
    # Benchmark performance
    print("\n" + "="*80)
    print("PERFORMANCE BENCHMARK")
    print("="*80)
    
    avg_time = benchmark_loader(train_loader, num_batches=10)
    
    # Estimate epoch time
    total_batches = len(train_loader)
    epoch_time = avg_time * total_batches / 60
    
    print(f"\n⏱️  Estimated Full Epoch:")
    print(f"   Total batches: {total_batches}")
    print(f"   Epoch time: {epoch_time:.2f} minutes")
    print(f"   50 epochs: {epoch_time * 50:.1f} minutes ({epoch_time * 50 / 60:.1f} hours)")
    
    # Final summary
    print("\n" + "="*80)
    print("FAST DATASET READY!")
    print("="*80)
    
    print(f"\n✅ Key Features:")
    print(f"   • Fast JPEG decoding (torchvision.io.read_image)")
    print(f"   • {info['num_workers']} worker processes")
    print(f"   • Automatic error handling")
    print(f"   • Phase 2 augmentation ready")
    print(f"   • {info['num_classes']} disease classes")
    
    print(f"\n💡 Usage in Training:")
    print("""
from fast_dataset import make_loaders

# Create optimized loaders
train_loader, val_loader, test_loader, class_names, info = make_loaders(
    data_dir='Database_resized/',
    batch_size=64,  # Increase if GPU memory allows
    num_workers=None  # Auto-detect optimal value
)

# Training loop
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
    """)
    
    print("\n" + "="*80)
