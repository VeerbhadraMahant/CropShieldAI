"""
Data Transforms for CropShield AI
==================================
Provides modular transform pipelines for training and validation.

Features:
- Conservative, Moderate, and Aggressive augmentation presets
- ImageNet normalization (compatible with transfer learning)
- Preserves disease color/texture features
- Pillow-SIMD compatible (efficient geometric operations)
- Float32 output tensors

Usage:
    from transforms import get_transforms
    
    # Get moderate augmentation (recommended starting point)
    train_tfm, val_tfm = get_transforms(mode='moderate')
    
    # Apply to PIL image
    image = Image.open('sample.jpg')
    tensor = train_tfm(image)  # Shape: [3, 224, 224], dtype: float32
"""

import torch
import torchvision.transforms as T
from typing import Tuple, Optional


# ImageNet normalization constants
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# Target image size
IMAGE_SIZE = 224
RESIZE_SIZE = 256  # Slightly larger for cropping


def get_transforms(
    mode: str = 'moderate',
    image_size: int = IMAGE_SIZE,
    mean: Optional[list] = None,
    std: Optional[list] = None
) -> Tuple[T.Compose, T.Compose]:
    """
    Get training and validation transform pipelines.
    
    Args:
        mode: Augmentation strength ('conservative', 'moderate', 'aggressive')
        image_size: Target image size (default: 224)
        mean: Normalization mean (default: ImageNet)
        std: Normalization std (default: ImageNet)
    
    Returns:
        (train_transforms, val_transforms) tuple
    
    Modes:
        - conservative: Minimal augmentation, safe for initial training
        - moderate: Balanced augmentation, recommended for most cases
        - aggressive: Heavy augmentation, use if validation shows underfitting
    
    Example:
        >>> train_tfm, val_tfm = get_transforms('moderate')
        >>> train_tfm(image).shape
        torch.Size([3, 224, 224])
    """
    if mean is None:
        mean = IMAGENET_MEAN
    if std is None:
        std = IMAGENET_STD
    
    if mode == 'conservative':
        train_transforms = get_conservative_transforms(image_size, mean, std)
    elif mode == 'moderate':
        train_transforms = get_moderate_transforms(image_size, mean, std)
    elif mode == 'aggressive':
        train_transforms = get_aggressive_transforms(image_size, mean, std)
    else:
        raise ValueError(f"Unknown mode: {mode}. Choose 'conservative', 'moderate', or 'aggressive'")
    
    val_transforms = get_validation_transforms(image_size, mean, std)
    
    return train_transforms, val_transforms


def get_conservative_transforms(
    image_size: int = IMAGE_SIZE,
    mean: list = IMAGENET_MEAN,
    std: list = IMAGENET_STD
) -> T.Compose:
    """
    Conservative augmentation pipeline.
    
    Augmentations:
    - Random crop with scale 0.8-1.0
    - Horizontal flip (50% probability)
    - Subtle color jitter
    
    Use Case: Initial training, uncertain about dataset
    """
    return T.Compose([
        T.RandomResizedCrop(
            image_size,
            scale=(0.8, 1.0),
            ratio=(0.9, 1.1),
            interpolation=T.InterpolationMode.BILINEAR
        ),
        T.RandomHorizontalFlip(p=0.5),
        T.ColorJitter(
            brightness=0.1,
            contrast=0.1,
            saturation=0.1,
            hue=0.02
        ),
        T.ToTensor(),  # Converts to [0, 1] float32
        T.Normalize(mean=mean, std=std)
    ])


def get_moderate_transforms(
    image_size: int = IMAGE_SIZE,
    mean: list = IMAGENET_MEAN,
    std: list = IMAGENET_STD
) -> T.Compose:
    """
    Moderate augmentation pipeline (RECOMMENDED).
    
    Augmentations:
    - Random crop with scale 0.7-1.0
    - Horizontal flip (50%)
    - Vertical flip (30%)
    - Random rotation ±15°
    - Moderate color jitter
    - Random erasing (10%)
    
    Use Case: Balanced generalization, suitable for most plant disease datasets
    """
    return T.Compose([
        T.RandomResizedCrop(
            image_size,
            scale=(0.7, 1.0),
            ratio=(0.85, 1.15),
            interpolation=T.InterpolationMode.BILINEAR
        ),
        T.RandomHorizontalFlip(p=0.5),
        T.RandomVerticalFlip(p=0.3),
        T.RandomRotation(
            degrees=15,
            interpolation=T.InterpolationMode.BILINEAR,
            fill=0
        ),
        T.ColorJitter(
            brightness=0.2,
            contrast=0.2,
            saturation=0.15,
            hue=0.03
        ),
        T.ToTensor(),  # Converts to [0, 1] float32
        T.RandomErasing(
            p=0.1,
            scale=(0.02, 0.15),
            ratio=(0.3, 3.3),
            value='random'
        ),
        T.Normalize(mean=mean, std=std)
    ])


def get_aggressive_transforms(
    image_size: int = IMAGE_SIZE,
    mean: list = IMAGENET_MEAN,
    std: list = IMAGENET_STD
) -> T.Compose:
    """
    Aggressive augmentation pipeline.
    
    Augmentations:
    - Random crop with scale 0.6-1.0
    - Horizontal flip (50%)
    - Vertical flip (50%)
    - Random rotation ±20°
    - Strong color jitter
    - Random erasing (20%)
    - Gaussian blur (10%)
    
    Use Case: Large dataset, need strong regularization, validation shows underfitting
    
    Warning: May harm performance if dataset is small or validation already shows overfitting
    """
    return T.Compose([
        T.RandomResizedCrop(
            image_size,
            scale=(0.6, 1.0),
            ratio=(0.8, 1.2),
            interpolation=T.InterpolationMode.BILINEAR
        ),
        T.RandomHorizontalFlip(p=0.5),
        T.RandomVerticalFlip(p=0.5),
        T.RandomRotation(
            degrees=20,
            interpolation=T.InterpolationMode.BILINEAR,
            fill=0
        ),
        T.ColorJitter(
            brightness=0.3,
            contrast=0.3,
            saturation=0.2,
            hue=0.05
        ),
        T.GaussianBlur(kernel_size=5, sigma=(0.1, 2.0)),
        T.ToTensor(),  # Converts to [0, 1] float32
        T.RandomErasing(
            p=0.2,
            scale=(0.02, 0.2),
            ratio=(0.3, 3.3),
            value='random'
        ),
        T.Normalize(mean=mean, std=std)
    ])


def get_validation_transforms(
    image_size: int = IMAGE_SIZE,
    mean: list = IMAGENET_MEAN,
    std: list = IMAGENET_STD
) -> T.Compose:
    """
    Validation/Test transform pipeline.
    
    No augmentation, only preprocessing:
    - Resize to 256
    - Center crop to 224
    - Normalize to ImageNet stats
    
    Use Case: Validation, test, and inference
    """
    resize_size = int(image_size * 256 / 224)  # Maintain aspect ratio
    return T.Compose([
        T.Resize(resize_size, interpolation=T.InterpolationMode.BILINEAR),
        T.CenterCrop(image_size),
        T.ToTensor(),  # Converts to [0, 1] float32
        T.Normalize(mean=mean, std=std)
    ])


def denormalize(tensor: torch.Tensor, mean: list = IMAGENET_MEAN, std: list = IMAGENET_STD) -> torch.Tensor:
    """
    Denormalize a tensor for visualization.
    
    Args:
        tensor: Normalized tensor [C, H, W] or [B, C, H, W]
        mean: Normalization mean used
        std: Normalization std used
    
    Returns:
        Denormalized tensor in [0, 1] range
    
    Example:
        >>> normalized = train_tfm(image)
        >>> original = denormalize(normalized)
        >>> plt.imshow(original.permute(1, 2, 0))
    """
    mean = torch.tensor(mean).view(-1, 1, 1)
    std = torch.tensor(std).view(-1, 1, 1)
    
    # Handle batch dimension
    if tensor.dim() == 4:
        mean = mean.unsqueeze(0)
        std = std.unsqueeze(0)
    
    denorm = tensor * std + mean
    return torch.clamp(denorm, 0, 1)


def visualize_augmentations(
    image_path: str,
    mode: str = 'moderate',
    num_samples: int = 8,
    save_path: Optional[str] = None
):
    """
    Visualize augmentation effects on a single image.
    
    Args:
        image_path: Path to input image
        mode: Augmentation mode
        num_samples: Number of augmented samples to generate
        save_path: Optional path to save visualization
    
    Example:
        >>> visualize_augmentations('sample.jpg', mode='moderate', num_samples=8)
    """
    from PIL import Image
    import matplotlib.pyplot as plt
    
    # Load image
    image = Image.open(image_path).convert('RGB')
    
    # Get transforms
    train_tfm, _ = get_transforms(mode)
    
    # Generate augmented samples
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    axes = axes.flatten()
    
    for i, ax in enumerate(axes):
        if i < num_samples:
            # Apply augmentation
            augmented = train_tfm(image)
            
            # Denormalize for visualization
            denorm = denormalize(augmented)
            
            # Convert to numpy and permute to HWC
            img_np = denorm.permute(1, 2, 0).numpy()
            
            ax.imshow(img_np)
            ax.set_title(f'Augmented {i+1}')
            ax.axis('off')
        else:
            ax.axis('off')
    
    plt.suptitle(f'Augmentation Mode: {mode.upper()}', fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"✅ Saved visualization to {save_path}")
    else:
        plt.show()


# Example usage and testing
if __name__ == '__main__':
    import numpy as np
    from PIL import Image
    
    print("=" * 70)
    print("CropShield AI - Transform Pipeline Testing")
    print("=" * 70)
    
    # Test all modes
    modes = ['conservative', 'moderate', 'aggressive']
    
    for mode in modes:
        print(f"\n📋 Testing {mode.upper()} mode:")
        train_tfm, val_tfm = get_transforms(mode)
        
        # Create dummy image
        dummy_image = Image.fromarray(np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8))
        
        # Apply transforms
        train_tensor = train_tfm(dummy_image)
        val_tensor = val_tfm(dummy_image)
        
        # Validate output
        print(f"  ✅ Training transform output: shape={train_tensor.shape}, dtype={train_tensor.dtype}")
        print(f"     Range: [{train_tensor.min():.3f}, {train_tensor.max():.3f}]")
        print(f"  ✅ Validation transform output: shape={val_tensor.shape}, dtype={val_tensor.dtype}")
        print(f"     Range: [{val_tensor.min():.3f}, {val_tensor.max():.3f}]")
        
        # Check float32
        assert train_tensor.dtype == torch.float32, "Training tensor must be float32"
        assert val_tensor.dtype == torch.float32, "Validation tensor must be float32"
        
        # Check shape
        assert train_tensor.shape == (3, 224, 224), "Training tensor must be [3, 224, 224]"
        assert val_tensor.shape == (3, 224, 224), "Validation tensor must be [3, 224, 224]"
    
    print("\n" + "=" * 70)
    print("✅ ALL TESTS PASSED")
    print("=" * 70)
    print("\nQuick Start:")
    print("  from transforms import get_transforms")
    print("  train_tfm, val_tfm = get_transforms('moderate')")
    print("  tensor = train_tfm(pil_image)")
