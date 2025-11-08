"""
CropShield AI - Pre-Training Diagnostic Check (Script Version)
Quick validation script to verify everything works before model training

Checks:
1. GPU availability and configuration
2. DataLoader functionality (FastImageFolder or WebDataset)
3. Tensor shapes, dtypes, and label mappings
4. Image visualization with class names
5. Loading performance and throughput
"""

import torch
import matplotlib.pyplot as plt
import numpy as np
import time
from pathlib import Path


def check_gpu():
    """Check GPU availability and configuration"""
    print("=" * 80)
    print("GPU DIAGNOSTICS")
    print("=" * 80)
    
    cuda_available = torch.cuda.is_available()
    print(f"\n🎮 CUDA Available: {cuda_available}")
    
    if cuda_available:
        print(f"   GPU Device: {torch.cuda.get_device_name(0)}")
        print(f"   CUDA Version: {torch.version.cuda}")
        print(f"   Current Device: cuda:{torch.cuda.current_device()}")
        
        total_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        allocated = torch.cuda.memory_allocated(0) / (1024**3)
        cached = torch.cuda.memory_reserved(0) / (1024**3)
        
        print(f"\n💾 GPU Memory:")
        print(f"   Total: {total_memory:.2f} GB")
        print(f"   Allocated: {allocated:.3f} GB")
        print(f"   Cached: {cached:.3f} GB")
        print(f"   Free: {total_memory - cached:.2f} GB")
        
        device = torch.device('cuda')
        print("\n✅ GPU ready for training!")
    else:
        device = torch.device('cpu')
        print("\n⚠️  GPU not available, using CPU")
    
    print(f"\n🎯 Using device: {device}")
    print("=" * 80)
    
    return cuda_available, device


def create_dataloader(use_webdataset=True, batch_size=32, num_workers=0):
    """Create DataLoader (WebDataset or FastImageFolder)"""
    print("\n📦 Creating DataLoader...")
    
    if use_webdataset:
        if not Path('shards/').exists():
            raise FileNotFoundError(
                "shards/ directory not found. "
                "Run: python scripts/create_webdataset_shards.py"
            )
        
        from webdataset_loader import make_webdataset_loaders
        
        train_loader, val_loader, test_loader, class_info = make_webdataset_loaders(
            shards_dir='shards/',
            batch_size=batch_size,
            num_workers=num_workers
        )
        
        classes = class_info['classes']
        num_classes = class_info['num_classes']
        
    else:
        if not Path('Database_resized/').exists():
            raise FileNotFoundError(
                "Database_resized/ directory not found. "
                "Run: python scripts/resize_images.py"
            )
        
        from fast_dataset import make_loaders
        
        train_loader, val_loader, test_loader, class_to_idx, classes = make_loaders(
            data_dir='Database_resized/',
            batch_size=batch_size,
            num_workers=num_workers
        )
        
        num_classes = len(classes)
    
    print(f"✅ DataLoader created successfully!")
    print(f"\n📊 Dataset Statistics:")
    print(f"   Number of classes: {num_classes}")
    print(f"   Classes (first 5): {classes[:5]}")
    
    return train_loader, val_loader, test_loader, classes, num_classes


def analyze_batch(images, labels, classes):
    """Analyze a single batch"""
    print("\n" + "=" * 80)
    print("BATCH ANALYSIS")
    print("=" * 80)
    
    print(f"\n📦 Tensor Information:")
    print(f"   Images shape: {images.shape}")
    print(f"   Images dtype: {images.dtype}")
    print(f"   Labels shape: {labels.shape}")
    print(f"   Labels dtype: {labels.dtype}")
    
    print(f"\n📊 Value Ranges:")
    print(f"   Image min: {images.min():.3f}")
    print(f"   Image max: {images.max():.3f}")
    print(f"   Image mean: {images.mean():.3f}")
    print(f"   Image std: {images.std():.3f}")
    
    print(f"\n🏷️  Label Information:")
    print(f"   Label min: {labels.min()}")
    print(f"   Label max: {labels.max()}")
    print(f"   Unique labels: {torch.unique(labels).tolist()}")
    
    print("\n" + "=" * 80)
    print("LABEL MAPPING (First 8 samples)")
    print("=" * 80)
    
    for i in range(min(8, len(labels))):
        label_idx = labels[i].item()
        class_name = classes[label_idx]
        print(f"   Sample {i+1}: Label {label_idx:2d} → {class_name}")
    
    print("=" * 80)


def denormalize(tensor):
    """Denormalize image tensor from ImageNet normalization"""
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    img = tensor * std + mean
    img = torch.clamp(img, 0, 1)
    return img


def visualize_batch(images, labels, classes, num_images=8):
    """Display a grid of images with class names"""
    print("\n🖼️  Displaying sample images...")
    
    num_images = min(num_images, len(images))
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    axes = axes.flatten()
    
    for i in range(num_images):
        img = images[i]
        label_idx = labels[i].item()
        class_name = classes[label_idx]
        
        img = denormalize(img)
        img_np = img.cpu().numpy().transpose(1, 2, 0)
        
        axes[i].imshow(img_np)
        axes[i].set_title(f"{class_name}\n(Label: {label_idx})", fontsize=10)
        axes[i].axis('off')
    
    plt.suptitle('Sample Images from Training Batch', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig('diagnostic_sample_images.png', dpi=150, bbox_inches='tight')
    print("✅ Sample images saved to: diagnostic_sample_images.png")
    plt.show()


def test_gpu_transfer(images, labels, device):
    """Test GPU transfer speed"""
    print("\n" + "=" * 80)
    print("GPU TRANSFER TEST")
    print("=" * 80)
    
    if device.type == 'cuda':
        print("\n⏱️  Testing GPU transfer speed...")
        
        start_time = time.time()
        images_gpu = images.to(device, non_blocking=True)
        labels_gpu = labels.to(device, non_blocking=True)
        torch.cuda.synchronize()
        transfer_time = time.time() - start_time
        
        print(f"✅ Transfer complete in {transfer_time*1000:.2f}ms")
        
        allocated = torch.cuda.memory_allocated(0) / (1024**3)
        print(f"\n💾 GPU Memory Allocated: {allocated:.3f} GB")
        
        batch_size_mb = (images.numel() * images.element_size()) / (1024**2)
        transfer_speed = batch_size_mb / transfer_time
        print(f"🚀 Transfer Speed: {transfer_speed:.1f} MB/s")
        
        del images_gpu, labels_gpu
        torch.cuda.empty_cache()
    else:
        print("\n⚠️  No GPU available, skipping transfer test")
    
    print("=" * 80)


def test_multi_batch(train_loader, device, batch_size, num_batches=10):
    """Test loading multiple batches"""
    print("\n" + "=" * 80)
    print(f"MULTI-BATCH LOADING TEST ({num_batches} batches)")
    print("=" * 80)
    
    start_time = time.time()
    batch_times = []
    total_images = 0
    
    for i, (batch_images, batch_labels) in enumerate(train_loader):
        batch_start = time.time()
        
        if device.type == 'cuda':
            batch_images = batch_images.to(device, non_blocking=True)
            batch_labels = batch_labels.to(device, non_blocking=True)
        
        batch_time = time.time() - batch_start
        batch_times.append(batch_time)
        total_images += batch_images.size(0)
        
        if i + 1 >= num_batches:
            break
    
    total_time = time.time() - start_time
    avg_batch_time = np.mean(batch_times)
    throughput = total_images / total_time
    
    print(f"\n✅ Loaded {num_batches} batches ({total_images} images)")
    print(f"\n📊 Loading Statistics:")
    print(f"   Avg batch time: {avg_batch_time*1000:.1f}ms")
    print(f"   Throughput: {throughput:.1f} images/second")
    
    train_samples = 17901
    epoch_time = train_samples / throughput
    print(f"\n⏱️  Estimated Training Time:")
    print(f"   Time per epoch: {epoch_time/60:.2f} minutes")
    print(f"   Time for 50 epochs: {epoch_time*50/60:.1f} minutes")
    
    print("=" * 80)
    
    return throughput, epoch_time


def print_summary(cuda_available, use_webdataset, batch_size, num_workers, 
                  num_classes, throughput, epoch_time):
    """Print final diagnostic summary"""
    print("\n" + "=" * 80)
    print("DIAGNOSTIC SUMMARY")
    print("=" * 80)
    
    checks = [
        ("✅", "PyTorch installed", f"v{torch.__version__}"),
        ("✅" if cuda_available else "⚠️ ", "GPU available", 
         f"{torch.cuda.get_device_name(0)}" if cuda_available else "CPU only"),
        ("✅", "DataLoader created", f"{'WebDataset' if use_webdataset else 'FastImageFolder'}"),
        ("✅", "Batch loading works", f"{batch_size} images/batch"),
        ("✅", "Throughput measured", f"{throughput:.1f} img/s"),
    ]
    
    print("\n📋 System Checks:")
    for status, check, detail in checks:
        print(f"   {status} {check:<30} {detail}")
    
    print("\n📊 Configuration:")
    print(f"   Classes: {num_classes}")
    print(f"   Batch size: {batch_size}")
    print(f"   Num workers: {num_workers}")
    print(f"   Data loading: {throughput:.1f} img/s")
    print(f"   Epoch time: ~{epoch_time/60:.2f} minutes")
    
    print("\n" + "=" * 80)
    print("🎉 ALL DIAGNOSTICS PASSED!")
    print("=" * 80)
    
    print("\n✅ Ready for CNN Model Training!")
    print("\nNext steps:")
    print("   1. Build CNN model architecture")
    print("   2. Create training script")
    print("   3. Add mixed precision training (AMP)")
    print("   4. Implement learning rate scheduling")
    print("   5. Monitor with TensorBoard")
    print("\n🚀 Let's build that model!")


def main():
    """Main diagnostic workflow"""
    print("=" * 80)
    print("CROPSHIELD AI - PRE-TRAINING DIAGNOSTIC CHECK")
    print("=" * 80)
    
    # Configuration
    USE_WEBDATASET = True
    BATCH_SIZE = 32
    NUM_WORKERS = 0  # Use 0 for Windows
    
    print(f"\n⚙️  Configuration:")
    print(f"   Loader: {'WebDataset' if USE_WEBDATASET else 'FastImageFolder'}")
    print(f"   Batch size: {BATCH_SIZE}")
    print(f"   Num workers: {NUM_WORKERS}")
    
    # 1. Check GPU
    cuda_available, device = check_gpu()
    
    # 2. Create DataLoader
    train_loader, val_loader, test_loader, classes, num_classes = create_dataloader(
        USE_WEBDATASET, BATCH_SIZE, NUM_WORKERS
    )
    
    # 3. Load and analyze one batch
    print("\n⏱️  Loading first batch...")
    start_time = time.time()
    images, labels = next(iter(train_loader))
    load_time = time.time() - start_time
    print(f"✅ Batch loaded in {load_time:.3f} seconds")
    
    analyze_batch(images, labels, classes)
    
    # 4. Visualize images
    visualize_batch(images, labels, classes)
    
    # 5. Test GPU transfer
    test_gpu_transfer(images, labels, device)
    
    # 6. Test multi-batch loading
    throughput, epoch_time = test_multi_batch(train_loader, device, BATCH_SIZE)
    
    # 7. Print summary
    print_summary(
        cuda_available, USE_WEBDATASET, BATCH_SIZE, NUM_WORKERS,
        num_classes, throughput, epoch_time
    )


if __name__ == "__main__":
    main()
