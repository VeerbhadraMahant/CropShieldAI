"""
Batch Visualization Tool for DataLoader
Visualize augmented training batches to confirm realistic transformations
"""

import torch
import matplotlib.pyplot as plt
import numpy as np
from typing import List, Tuple, Optional


def denormalize(tensor: torch.Tensor, 
                mean: List[float] = [0.485, 0.456, 0.406],
                std: List[float] = [0.229, 0.224, 0.225]) -> torch.Tensor:
    """
    Reverse ImageNet normalization to get displayable images.
    
    Args:
        tensor: Normalized image tensor [C, H, W] or [B, C, H, W]
        mean: Mean values used in normalization
        std: Std values used in normalization
        
    Returns:
        Denormalized tensor with values in [0, 1]
    """
    # Clone to avoid modifying original
    tensor = tensor.clone()
    
    # Convert lists to tensors
    mean = torch.tensor(mean).view(-1, 1, 1)
    std = torch.tensor(std).view(-1, 1, 1)
    
    # Handle batch dimension
    if tensor.dim() == 4:  # [B, C, H, W]
        mean = mean.unsqueeze(0)
        std = std.unsqueeze(0)
    
    # Reverse normalization: x_original = x_normalized * std + mean
    tensor = tensor * std + mean
    
    # Clip to valid range [0, 1]
    tensor = torch.clamp(tensor, 0, 1)
    
    return tensor


def visualize_batch(loader, class_names: List[str], 
                   num_images: int = 10,
                   figsize: Tuple[int, int] = (20, 10),
                   save_path: Optional[str] = None,
                   title: str = "Training Batch with Augmentations"):
    """
    Visualize a batch of images from the DataLoader with class labels.
    
    Args:
        loader: PyTorch DataLoader (train_loader or val_loader)
        class_names: List of class names for labeling
        num_images: Number of images to display (default: 10)
        figsize: Figure size as (width, height) in inches
        save_path: Optional path to save the visualization
        title: Title for the plot
        
    Returns:
        fig, axes: Matplotlib figure and axes objects
    """
    # Get one batch
    images, labels = next(iter(loader))
    
    # Limit to num_images
    num_images = min(num_images, images.size(0))
    images = images[:num_images]
    labels = labels[:num_images]
    
    # Denormalize images
    images_denorm = denormalize(images)
    
    # Calculate grid dimensions
    cols = 5
    rows = (num_images + cols - 1) // cols
    
    # Create figure
    fig, axes = plt.subplots(rows, cols, figsize=figsize)
    axes = axes.flatten() if rows > 1 else [axes] if cols == 1 else axes
    
    # Plot each image
    for idx in range(num_images):
        # Convert tensor to numpy: [C, H, W] -> [H, W, C]
        img = images_denorm[idx].permute(1, 2, 0).cpu().numpy()
        label_idx = labels[idx].item()
        class_name = class_names[label_idx]
        
        # Display image
        axes[idx].imshow(img)
        axes[idx].set_title(f"{class_name}\n(Class {label_idx})", 
                           fontsize=10, fontweight='bold')
        axes[idx].axis('off')
    
    # Hide unused subplots
    for idx in range(num_images, len(axes)):
        axes[idx].axis('off')
    
    # Overall title
    fig.suptitle(title, fontsize=16, fontweight='bold', y=0.98)
    plt.tight_layout()
    
    # Save if path provided
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"✅ Visualization saved to: {save_path}")
    
    return fig, axes


def visualize_augmentation_comparison(loader, class_names: List[str],
                                     num_samples: int = 3,
                                     figsize: Tuple[int, int] = (20, 12),
                                     save_path: Optional[str] = None):
    """
    Compare multiple augmented versions of the same images.
    Fetches multiple batches to show augmentation variety.
    
    Args:
        loader: PyTorch DataLoader (train_loader)
        class_names: List of class names
        num_samples: Number of unique samples to show (default: 3)
        figsize: Figure size
        save_path: Optional save path
        
    Returns:
        fig, axes: Matplotlib figure and axes
    """
    # Get multiple batches to find repeated augmentations
    # (This assumes you're fetching from same dataset)
    batches = []
    for i, (images, labels) in enumerate(loader):
        batches.append((images[:num_samples], labels[:num_samples]))
        if i >= 3:  # Get 4 batches
            break
    
    # Create grid: num_samples rows x 4 columns (4 different augmentations)
    fig, axes = plt.subplots(num_samples, 4, figsize=figsize)
    if num_samples == 1:
        axes = axes.reshape(1, -1)
    
    for sample_idx in range(num_samples):
        for aug_idx, (images, labels) in enumerate(batches):
            if sample_idx < images.size(0):
                # Denormalize
                img_denorm = denormalize(images[sample_idx])
                img_np = img_denorm.permute(1, 2, 0).cpu().numpy()
                
                # Get label
                label_idx = labels[sample_idx].item()
                class_name = class_names[label_idx]
                
                # Plot
                axes[sample_idx, aug_idx].imshow(img_np)
                if aug_idx == 0:
                    axes[sample_idx, aug_idx].set_ylabel(
                        f"{class_name}", 
                        fontsize=12, 
                        fontweight='bold',
                        rotation=0,
                        labelpad=80,
                        va='center'
                    )
                axes[sample_idx, aug_idx].set_title(f"Aug {aug_idx + 1}", fontsize=10)
                axes[sample_idx, aug_idx].axis('off')
    
    fig.suptitle("Augmentation Variety: Same Images, Different Transforms", 
                fontsize=16, fontweight='bold', y=0.98)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"✅ Comparison saved to: {save_path}")
    
    return fig, axes


def check_augmentation_statistics(loader, num_batches: int = 5):
    """
    Print statistics about augmented batches to verify they look reasonable.
    
    Args:
        loader: DataLoader to analyze
        num_batches: Number of batches to sample
    """
    print("\n" + "="*60)
    print("AUGMENTATION STATISTICS CHECK")
    print("="*60)
    
    all_images = []
    all_labels = []
    
    for i, (images, labels) in enumerate(loader):
        if i >= num_batches:
            break
        all_images.append(images)
        all_labels.append(labels)
    
    all_images = torch.cat(all_images, dim=0)
    all_labels = torch.cat(all_labels, dim=0)
    
    # Denormalize for realistic statistics
    images_denorm = denormalize(all_images)
    
    print(f"\n📊 Analyzed {all_images.size(0)} images from {num_batches} batches")
    print(f"   Batch size: {loader.batch_size}")
    
    print("\n🎨 Pixel Value Statistics (after denormalization):")
    print(f"   Min:  {images_denorm.min().item():.4f}")
    print(f"   Max:  {images_denorm.max().item():.4f}")
    print(f"   Mean: {images_denorm.mean().item():.4f}")
    print(f"   Std:  {images_denorm.std().item():.4f}")
    
    print("\n🌈 Per-Channel Statistics:")
    for c, channel in enumerate(['Red', 'Green', 'Blue']):
        channel_data = images_denorm[:, c, :, :]
        print(f"   {channel:6s}: mean={channel_data.mean():.4f}, "
              f"std={channel_data.std():.4f}, "
              f"min={channel_data.min():.4f}, "
              f"max={channel_data.max():.4f}")
    
    print("\n🏷️  Label Distribution:")
    unique, counts = torch.unique(all_labels, return_counts=True)
    print(f"   Unique classes: {len(unique)}")
    print(f"   Samples per class: {counts.tolist()[:10]}..." if len(counts) > 10 
          else f"   Samples per class: {counts.tolist()}")
    
    print("\n✅ Checks:")
    checks_passed = 0
    total_checks = 4
    
    # Check 1: Values in [0, 1]
    if 0 <= images_denorm.min() and images_denorm.max() <= 1.05:  # Small tolerance
        print("   ✅ Denormalized values in valid range [0, 1]")
        checks_passed += 1
    else:
        print("   ❌ Denormalized values out of range!")
    
    # Check 2: Mean around 0.5
    mean = images_denorm.mean().item()
    if 0.3 <= mean <= 0.7:
        print(f"   ✅ Mean ({mean:.3f}) is reasonable for natural images")
        checks_passed += 1
    else:
        print(f"   ⚠️  Mean ({mean:.3f}) seems unusual")
    
    # Check 3: Std reasonable
    std = images_denorm.std().item()
    if 0.15 <= std <= 0.35:
        print(f"   ✅ Std ({std:.3f}) shows good variety")
        checks_passed += 1
    else:
        print(f"   ⚠️  Std ({std:.3f}) might indicate issue")
    
    # Check 4: Multiple classes present
    if len(unique) > 1:
        print(f"   ✅ Multiple classes present ({len(unique)} classes)")
        checks_passed += 1
    else:
        print("   ❌ Only one class in batch!")
    
    print(f"\n📊 Summary: {checks_passed}/{total_checks} checks passed")
    print("="*60 + "\n")


def quick_visual_check(train_loader, val_loader, class_names: List[str]):
    """
    Quick visual check of both train and validation loaders.
    
    Args:
        train_loader: Training DataLoader
        val_loader: Validation DataLoader
        class_names: List of class names
    """
    print("\n" + "="*60)
    print("QUICK VISUAL CHECK")
    print("="*60)
    
    # Training batch
    print("\n1️⃣  Visualizing TRAINING batch (with augmentations)...")
    fig1, _ = visualize_batch(
        train_loader, 
        class_names, 
        num_images=10,
        title="Training Batch with Augmentations",
        save_path="train_batch_visualization.png"
    )
    plt.show()
    
    # Validation batch
    print("\n2️⃣  Visualizing VALIDATION batch (no augmentations)...")
    fig2, _ = visualize_batch(
        val_loader, 
        class_names, 
        num_images=10,
        title="Validation Batch (No Augmentations)",
        save_path="val_batch_visualization.png"
    )
    plt.show()
    
    # Statistics
    print("\n3️⃣  Checking training batch statistics...")
    check_augmentation_statistics(train_loader, num_batches=5)
    
    print("✅ Visual check complete!")
    print("="*60 + "\n")


# Example usage
if __name__ == "__main__":
    """
    Example: How to use the visualization functions
    """
    from fast_dataset import make_loaders
    
    print("Loading DataLoaders...")
    train_loader, val_loader, test_loader, class_names, info = make_loaders(
        data_dir='Database_resized/',
        batch_size=32,
        augmentation_mode='moderate'
    )
    
    print(f"\n✅ Loaded {info['num_classes']} classes")
    print(f"   Training: {info['train_size']} images")
    print(f"   Validation: {info['val_size']} images")
    print(f"   Test: {info['test_size']} images")
    
    # Option 1: Quick check of both loaders
    quick_visual_check(train_loader, val_loader, class_names)
    
    # Option 2: Just training batch
    print("\n" + "="*60)
    print("SINGLE BATCH VISUALIZATION")
    print("="*60)
    visualize_batch(
        train_loader, 
        class_names, 
        num_images=10,
        save_path="training_batch_sample.png"
    )
    plt.show()
    
    # Option 3: Compare augmentation variety
    print("\n" + "="*60)
    print("AUGMENTATION VARIETY CHECK")
    print("="*60)
    visualize_augmentation_comparison(
        train_loader,
        class_names,
        num_samples=3,
        save_path="augmentation_comparison.png"
    )
    plt.show()
    
    print("\n✅ All visualizations complete!")
