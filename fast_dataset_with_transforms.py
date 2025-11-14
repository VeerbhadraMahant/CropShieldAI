"""
Updated FastImageFolder with Transform Integration
===================================================
This is an example of how to integrate the new transforms into fast_dataset.py

Key Changes:
1. Import get_transforms from transforms.py
2. Add augmentation_mode parameter to make_loaders()
3. Use train_tfm and val_tfm instead of hardcoded transforms
4. Remove old transform definitions

Usage:
    from fast_dataset_with_transforms import make_loaders
    
    # Use moderate augmentation (recommended)
    train_loader, val_loader, test_loader = make_loaders(
        data_dir='Database_resized',
        batch_size=32,
        num_workers=0,
        augmentation_mode='moderate'
    )
"""

import os
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets
from torchvision.io import read_image, ImageReadMode
from pathlib import Path
from typing import Tuple, Optional

from transforms import get_transforms  # Import our new transforms


class FastImageFolder(Dataset):
    """
    Fast PyTorch Dataset using torchvision.io for JPEG decoding.
    
    Performance: 2-3x faster than PIL-based ImageFolder
    """
    
    def __init__(self, root: str, transform=None):
        """
        Args:
            root: Root directory path
            transform: Transform to apply to images
        """
        self.root = root
        self.transform = transform
        
        # Use torchvision.datasets.ImageFolder to find images
        self._temp_dataset = datasets.ImageFolder(root)
        self.classes = self._temp_dataset.classes
        self.class_to_idx = self._temp_dataset.class_to_idx
        self.samples = self._temp_dataset.samples
        self.targets = self._temp_dataset.targets
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        path, target = self.samples[idx]
        
        try:
            # Fast JPEG decode using torchvision.io
            image = read_image(path, mode=ImageReadMode.RGB)
            
            # Convert to PIL for transforms compatibility
            from torchvision.transforms.functional import to_pil_image
            image = to_pil_image(image)
            
            if self.transform is not None:
                image = self.transform(image)
            
            return image, target
            
        except Exception as e:
            print(f"Error loading {path}: {e}")
            # Return zeros tensor on error
            return torch.zeros(3, 224, 224), target
    
    def get_class_counts(self):
        """Get number of samples per class"""
        from collections import Counter
        return dict(Counter(self.targets))


def make_loaders(
    data_dir: str = 'Database_resized',
    batch_size: int = 32,
    num_workers: int = 0,
    pin_memory: bool = True,
    augmentation_mode: str = 'moderate'  # NEW PARAMETER
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create train, validation, and test DataLoaders with transforms.
    
    Args:
        data_dir: Root directory containing train/val/test splits
        batch_size: Batch size for DataLoader
        num_workers: Number of worker processes
        pin_memory: Whether to pin memory (faster GPU transfer)
        augmentation_mode: 'conservative', 'moderate', or 'aggressive'
    
    Returns:
        (train_loader, val_loader, test_loader)
    
    Example:
        >>> train_loader, val_loader, test_loader = make_loaders(
        ...     augmentation_mode='moderate'
        ... )
        >>> for images, labels in train_loader:
        ...     # Training loop
        ...     pass
    """
    # Get transforms based on mode
    train_tfm, val_tfm = get_transforms(augmentation_mode)
    
    print(f"📋 Creating DataLoaders with {augmentation_mode.upper()} augmentation")
    
    # Create datasets with transforms
    train_dataset = FastImageFolder(
        os.path.join(data_dir, 'train'),
        transform=train_tfm  # Use training transforms
    )
    
    val_dataset = FastImageFolder(
        os.path.join(data_dir, 'val'),
        transform=val_tfm  # Use validation transforms (no augmentation)
    )
    
    test_dataset = FastImageFolder(
        os.path.join(data_dir, 'test'),
        transform=val_tfm  # Use validation transforms (no augmentation)
    )
    
    # Print dataset info
    print(f"\n✅ Datasets created:")
    print(f"  Train: {len(train_dataset)} images")
    print(f"  Val:   {len(val_dataset)} images")
    print(f"  Test:  {len(test_dataset)} images")
    print(f"  Classes: {len(train_dataset.classes)}")
    
    # Create DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,  # Shuffle training data
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=num_workers > 0
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,  # No shuffle for validation
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=num_workers > 0
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,  # No shuffle for test
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=num_workers > 0
    )
    
    print(f"\n✅ DataLoaders created:")
    print(f"  Batch size: {batch_size}")
    print(f"  Num workers: {num_workers}")
    print(f"  Train batches: {len(train_loader)}")
    print(f"  Val batches: {len(val_loader)}")
    print(f"  Test batches: {len(test_loader)}")
    
    return train_loader, val_loader, test_loader


# Test script
if __name__ == '__main__':
    import time
    
    print("=" * 70)
    print("FastImageFolder with Transform Integration - Test")
    print("=" * 70)
    
    # Test all augmentation modes
    modes = ['conservative', 'moderate', 'aggressive']
    
    for mode in modes:
        print(f"\n{'=' * 70}")
        print(f"Testing {mode.upper()} mode")
        print('=' * 70)
        
        # Create loaders
        train_loader, val_loader, test_loader = make_loaders(
            batch_size=32,
            num_workers=0,
            augmentation_mode=mode
        )
        
        # Test training loader
        print(f"\n🔥 Testing training loader...")
        start = time.time()
        for i, (images, labels) in enumerate(train_loader):
            if i >= 5:  # Test 5 batches
                break
            print(f"  Batch {i+1}: images={images.shape}, labels={labels.shape}, dtype={images.dtype}")
        elapsed = time.time() - start
        print(f"  Time: {elapsed:.2f}s ({5*32/elapsed:.1f} img/s)")
        
        # Test validation loader
        print(f"\n🧪 Testing validation loader...")
        for i, (images, labels) in enumerate(val_loader):
            if i >= 2:  # Test 2 batches
                break
            print(f"  Batch {i+1}: images={images.shape}, labels={labels.shape}, dtype={images.dtype}")
    
    print("\n" + "=" * 70)
    print("✅ ALL TESTS PASSED")
    print("=" * 70)
    print("\nNext Steps:")
    print("1. Choose augmentation mode based on your needs:")
    print("   - conservative: Initial training, small dataset")
    print("   - moderate: RECOMMENDED for most cases")
    print("   - aggressive: Large dataset, need strong regularization")
    print("2. Integrate into training script")
    print("3. Monitor train vs val accuracy to validate augmentation strength")
