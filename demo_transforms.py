"""
Transform Pipeline Demonstration
=================================
Shows how to apply transforms to actual dataset images.

Usage:
    python demo_transforms.py
"""

import os
import random
from pathlib import Path
from PIL import Image
import matplotlib.pyplot as plt
import torch

from transforms import get_transforms, denormalize, IMAGENET_MEAN, IMAGENET_STD


def find_sample_images(dataset_path: str = 'Database_resized', num_samples: int = 3):
    """
    Find random sample images from different classes.
    
    Args:
        dataset_path: Path to dataset directory
        num_samples: Number of samples to find
    
    Returns:
        List of (image_path, class_name) tuples
    """
    dataset_path = Path(dataset_path)
    
    if not dataset_path.exists():
        print(f"❌ Dataset not found at {dataset_path}")
        print("   Please ensure Database_resized/ directory exists")
        return []
    
    # Get all class directories
    class_dirs = [d for d in dataset_path.iterdir() if d.is_dir()]
    
    if not class_dirs:
        print(f"❌ No class directories found in {dataset_path}")
        return []
    
    # Sample random classes
    selected_classes = random.sample(class_dirs, min(num_samples, len(class_dirs)))
    
    samples = []
    for class_dir in selected_classes:
        # Get all images in class
        images = list(class_dir.glob('*.jpg')) + list(class_dir.glob('*.png'))
        
        if images:
            # Pick random image
            image_path = random.choice(images)
            class_name = class_dir.name
            samples.append((str(image_path), class_name))
    
    return samples


def demonstrate_single_image(image_path: str, class_name: str, mode: str = 'moderate'):
    """
    Show original image alongside augmented versions.
    
    Args:
        image_path: Path to image file
        class_name: Class name for display
        mode: Augmentation mode
    """
    print(f"\n{'=' * 70}")
    print(f"Image: {Path(image_path).name}")
    print(f"Class: {class_name}")
    print(f"Mode: {mode.upper()}")
    print('=' * 70)
    
    # Load image
    image = Image.open(image_path).convert('RGB')
    print(f"✅ Loaded image: {image.size} pixels, {image.mode}")
    
    # Get transforms
    train_tfm, val_tfm = get_transforms(mode)
    
    # Apply transforms
    train_tensor = train_tfm(image)
    val_tensor = val_tfm(image)
    
    print(f"\n📊 Tensor Statistics:")
    print(f"  Training:")
    print(f"    Shape: {train_tensor.shape}")
    print(f"    Dtype: {train_tensor.dtype}")
    print(f"    Range: [{train_tensor.min():.3f}, {train_tensor.max():.3f}]")
    print(f"    Mean: {train_tensor.mean():.3f}")
    print(f"    Std: {train_tensor.std():.3f}")
    
    print(f"  Validation:")
    print(f"    Shape: {val_tensor.shape}")
    print(f"    Dtype: {val_tensor.dtype}")
    print(f"    Range: [{val_tensor.min():.3f}, {val_tensor.max():.3f}]")
    print(f"    Mean: {val_tensor.mean():.3f}")
    print(f"    Std: {val_tensor.std():.3f}")
    
    # Denormalize for visualization
    train_denorm = denormalize(train_tensor)
    val_denorm = denormalize(val_tensor)
    
    # Create visualization
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    
    # Row 1: Original + 3 training augmentations
    axes[0, 0].imshow(image)
    axes[0, 0].set_title('Original Image', fontweight='bold')
    axes[0, 0].axis('off')
    
    for i in range(1, 4):
        aug_tensor = train_tfm(image)
        aug_denorm = denormalize(aug_tensor)
        axes[0, i].imshow(aug_denorm.permute(1, 2, 0).numpy())
        axes[0, i].set_title(f'Train Aug {i}')
        axes[0, i].axis('off')
    
    # Row 2: 3 more training augmentations + validation
    for i in range(3):
        aug_tensor = train_tfm(image)
        aug_denorm = denormalize(aug_tensor)
        axes[1, i].imshow(aug_denorm.permute(1, 2, 0).numpy())
        axes[1, i].set_title(f'Train Aug {i+4}')
        axes[1, i].axis('off')
    
    axes[1, 3].imshow(val_denorm.permute(1, 2, 0).numpy())
    axes[1, 3].set_title('Validation (No Aug)', fontweight='bold', color='blue')
    axes[1, 3].axis('off')
    
    plt.suptitle(
        f'{class_name} | Mode: {mode.upper()}',
        fontsize=16,
        fontweight='bold'
    )
    plt.tight_layout()
    
    # Save
    save_path = f'transform_demo_{mode}_{class_name.replace(" ", "_")}.png'
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"\n✅ Saved visualization to {save_path}")
    plt.close()


def compare_all_modes(image_path: str, class_name: str):
    """
    Compare conservative, moderate, and aggressive augmentations side-by-side.
    
    Args:
        image_path: Path to image file
        class_name: Class name for display
    """
    print(f"\n{'=' * 70}")
    print(f"Comparing All Augmentation Modes")
    print(f"Image: {Path(image_path).name}")
    print(f"Class: {class_name}")
    print('=' * 70)
    
    # Load image
    image = Image.open(image_path).convert('RGB')
    
    modes = ['conservative', 'moderate', 'aggressive']
    
    fig, axes = plt.subplots(3, 4, figsize=(16, 12))
    
    for row, mode in enumerate(modes):
        train_tfm, _ = get_transforms(mode)
        
        for col in range(4):
            aug_tensor = train_tfm(image)
            aug_denorm = denormalize(aug_tensor)
            
            axes[row, col].imshow(aug_denorm.permute(1, 2, 0).numpy())
            
            if col == 0:
                axes[row, col].set_ylabel(mode.upper(), fontsize=12, fontweight='bold')
            
            if row == 0:
                axes[row, col].set_title(f'Sample {col+1}')
            
            axes[row, col].axis('off')
    
    plt.suptitle(
        f'Augmentation Mode Comparison | {class_name}',
        fontsize=16,
        fontweight='bold'
    )
    plt.tight_layout()
    
    save_path = f'transform_comparison_{class_name.replace(" ", "_")}.png'
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"\n✅ Saved comparison to {save_path}")
    plt.close()


def batch_processing_demo():
    """
    Demonstrate batch processing with transforms.
    """
    print(f"\n{'=' * 70}")
    print("Batch Processing Demo")
    print('=' * 70)
    
    # Find sample images
    samples = find_sample_images(num_samples=4)
    
    if not samples:
        print("❌ No sample images found")
        return
    
    # Get moderate transforms
    train_tfm, val_tfm = get_transforms('moderate')
    
    # Process batch
    train_batch = []
    val_batch = []
    
    for image_path, class_name in samples:
        image = Image.open(image_path).convert('RGB')
        train_batch.append(train_tfm(image))
        val_batch.append(val_tfm(image))
    
    # Stack into batch
    train_batch = torch.stack(train_batch)
    val_batch = torch.stack(val_batch)
    
    print(f"\n✅ Batch Processing Results:")
    print(f"  Training batch: {train_batch.shape}, dtype={train_batch.dtype}")
    print(f"  Validation batch: {val_batch.shape}, dtype={val_batch.dtype}")
    print(f"\n  Memory usage:")
    print(f"    Train: {train_batch.element_size() * train_batch.nelement() / 1024:.2f} KB")
    print(f"    Val: {val_batch.element_size() * val_batch.nelement() / 1024:.2f} KB")
    
    # Visualize batch
    fig, axes = plt.subplots(2, len(samples), figsize=(4*len(samples), 8))
    
    if len(samples) == 1:
        axes = axes.reshape(-1, 1)
    
    for i, (image_path, class_name) in enumerate(samples):
        # Training
        train_denorm = denormalize(train_batch[i])
        axes[0, i].imshow(train_denorm.permute(1, 2, 0).numpy())
        axes[0, i].set_title(f'Train: {class_name}')
        axes[0, i].axis('off')
        
        # Validation
        val_denorm = denormalize(val_batch[i])
        axes[1, i].imshow(val_denorm.permute(1, 2, 0).numpy())
        axes[1, i].set_title(f'Val: {class_name}')
        axes[1, i].axis('off')
    
    plt.suptitle('Batch Processing Demo', fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    save_path = 'batch_processing_demo.png'
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"\n✅ Saved batch visualization to {save_path}")
    plt.close()


def main():
    """
    Main demonstration script.
    """
    print("\n" + "=" * 70)
    print("CropShield AI - Transform Pipeline Demonstration")
    print("=" * 70)
    
    # Find sample images
    samples = find_sample_images(num_samples=3)
    
    if not samples:
        print("\n❌ No sample images found. Please ensure Database_resized/ exists.")
        return
    
    print(f"\n✅ Found {len(samples)} sample images")
    
    # Demo 1: Show moderate augmentation for each sample
    print("\n" + "-" * 70)
    print("Demo 1: Moderate Augmentation (Recommended)")
    print("-" * 70)
    
    for image_path, class_name in samples:
        demonstrate_single_image(image_path, class_name, mode='moderate')
    
    # Demo 2: Compare all modes for first sample
    print("\n" + "-" * 70)
    print("Demo 2: Compare All Modes")
    print("-" * 70)
    
    compare_all_modes(samples[0][0], samples[0][1])
    
    # Demo 3: Batch processing
    print("\n" + "-" * 70)
    print("Demo 3: Batch Processing")
    print("-" * 70)
    
    batch_processing_demo()
    
    print("\n" + "=" * 70)
    print("✅ ALL DEMONSTRATIONS COMPLETE")
    print("=" * 70)
    print("\nGenerated files:")
    print("  - transform_demo_*.png (individual augmentation demos)")
    print("  - transform_comparison_*.png (mode comparison)")
    print("  - batch_processing_demo.png (batch demo)")
    print("\nNext Steps:")
    print("  1. Review generated images to verify augmentations look realistic")
    print("  2. If augmentations are too weak/strong, adjust mode in get_transforms()")
    print("  3. Integrate transforms into data loaders (fast_dataset.py or webdataset_loader.py)")
    print("  4. Start model training with transforms applied")


if __name__ == '__main__':
    main()
