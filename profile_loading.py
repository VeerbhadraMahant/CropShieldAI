"""
CropShield AI - Data Loading Performance Profiler
Diagnoses bottlenecks in PyTorch dataset loading
"""

import torch
import time
import os
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import random

print("="*80)
print("CROPSHIELD AI - DATA LOADING PERFORMANCE PROFILER")
print("="*80)

# Check dataset
dataset_path = 'Database/'
if not os.path.exists(dataset_path):
    print(f"❌ Error: Dataset path '{dataset_path}' not found!")
    exit(1)

# Basic transform
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

print(f"\n📁 Loading dataset from: {dataset_path}")
dataset = datasets.ImageFolder(dataset_path, transform=transform)
print(f"✅ Total images: {len(dataset):,}")
print(f"✅ Number of classes: {len(dataset.classes)}")
print(f"✅ Sample classes: {dataset.classes[:5]}")

# ==============================================================================
# Test 1: Single Image Load Time
# ==============================================================================
print("\n" + "="*80)
print("TEST 1: SINGLE IMAGE LOAD TIME")
print("="*80)

print("\nLoading first image...")
start = time.time()
img, label = dataset[0]
first_load_time = time.time() - start
print(f"✅ First image load: {first_load_time*1000:.2f}ms")
print(f"   Image shape: {img.shape}")

print("\nLoading random image (index 1000)...")
start = time.time()
img, label = dataset[1000] if len(dataset) > 1000 else dataset[len(dataset)//2]
random_load_time = time.time() - start
print(f"✅ Random image load: {random_load_time*1000:.2f}ms")

# Benchmark interpretation
if first_load_time > 0.1:
    print("⚠️  WARNING: Image loading is slow (>100ms)")
    print("   → Likely disk I/O bottleneck (HDD or slow storage)")
elif first_load_time > 0.05:
    print("⚠️  MODERATE: Image loading is acceptable but could be faster")
else:
    print("✅ GOOD: Image loading is fast (<50ms)")

# ==============================================================================
# Test 2: Sequential vs Random Access
# ==============================================================================
print("\n" + "="*80)
print("TEST 2: SEQUENTIAL vs RANDOM ACCESS (100 images)")
print("="*80)

print("\nTesting sequential access...")
start = time.time()
for i in range(100):
    _ = dataset[i]
seq_time = time.time() - start
seq_per_image = seq_time / 100
print(f"✅ Sequential: {seq_time:.2f}s ({seq_per_image*1000:.1f}ms per image)")

print("\nTesting random access...")
indices = random.sample(range(len(dataset)), 100)
start = time.time()
for i in indices:
    _ = dataset[i]
random_time = time.time() - start
random_per_image = random_time / 100
print(f"✅ Random: {random_time:.2f}s ({random_per_image*1000:.1f}ms per image)")

# Interpretation
ratio = random_time / seq_time
print(f"\n📊 Random/Sequential ratio: {ratio:.2f}x")
if ratio > 2.0:
    print("⚠️  WARNING: Random access is significantly slower")
    print("   → You're likely using an HDD (high seek time)")
    print("   → Consider moving dataset to SSD or caching to RAM")
elif ratio > 1.5:
    print("⚠️  MODERATE: Random access overhead detected")
else:
    print("✅ GOOD: Storage has low seek time (likely SSD)")

# ==============================================================================
# Test 3: DataLoader with Different num_workers
# ==============================================================================
print("\n" + "="*80)
print("TEST 3: DATALOADER PERFORMANCE (batch_size=32)")
print("="*80)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"\n🖥️  Device: {device}")
if torch.cuda.is_available():
    print(f"   GPU: {torch.cuda.get_device_name(0)}")

results = {}
for num_workers in [0, 2, 4, 8]:
    print(f"\n🔄 Testing num_workers={num_workers}...")
    
    try:
        loader = DataLoader(
            dataset, 
            batch_size=32, 
            num_workers=num_workers,
            pin_memory=(device.type == 'cuda'),
            persistent_workers=(num_workers > 0),
            shuffle=False
        )
        
        start = time.time()
        batch_count = 0
        for i, (images, labels) in enumerate(loader):
            if device.type == 'cuda':
                images = images.to(device)
            batch_count += 1
            if i >= 9:  # Test first 10 batches
                break
        
        elapsed = time.time() - start
        per_batch = elapsed / batch_count
        images_per_sec = (32 * batch_count) / elapsed
        
        results[num_workers] = {
            'total_time': elapsed,
            'per_batch': per_batch,
            'images_per_sec': images_per_sec
        }
        
        print(f"   ✅ Total: {elapsed:.2f}s | Per batch: {per_batch*1000:.1f}ms | Throughput: {images_per_sec:.1f} img/s")
        
    except Exception as e:
        print(f"   ❌ Error with num_workers={num_workers}: {str(e)}")
        results[num_workers] = None

# Find optimal num_workers
print("\n" + "="*80)
print("ANALYSIS & RECOMMENDATIONS")
print("="*80)

valid_results = {k: v for k, v in results.items() if v is not None}
if valid_results:
    best_workers = min(valid_results.keys(), key=lambda k: valid_results[k]['total_time'])
    best_time = valid_results[best_workers]['total_time']
    
    print(f"\n🏆 OPTIMAL num_workers: {best_workers}")
    print(f"   Best throughput: {valid_results[best_workers]['images_per_sec']:.1f} images/sec")
    print(f"   Time for 10 batches: {best_time:.2f}s")
    
    # Calculate estimated epoch time
    total_batches = len(dataset) // 32
    estimated_epoch = (best_time / 10) * total_batches / 60
    print(f"\n⏱️  ESTIMATED FULL EPOCH TIME: {estimated_epoch:.1f} minutes")
    print(f"   (Based on {total_batches:,} batches with batch_size=32)")
    
    # Performance assessment
    print("\n📊 PERFORMANCE ASSESSMENT:")
    if estimated_epoch < 5:
        print("   ✅ EXCELLENT: Very fast loading")
    elif estimated_epoch < 10:
        print("   ✅ GOOD: Acceptable loading speed")
    elif estimated_epoch < 20:
        print("   ⚠️  MODERATE: Consider optimization")
    else:
        print("   ❌ SLOW: Significant optimization needed")
    
    # Speedup analysis
    if 0 in valid_results and best_workers != 0:
        speedup = valid_results[0]['total_time'] / valid_results[best_workers]['total_time']
        print(f"\n🚀 SPEEDUP from num_workers=0 to {best_workers}: {speedup:.2f}x faster")

# ==============================================================================
# Recommendations
# ==============================================================================
print("\n" + "="*80)
print("💡 OPTIMIZATION RECOMMENDATIONS")
print("="*80)

recommendations = []

# Based on single image load time
if first_load_time > 0.1:
    recommendations.append("1. 💾 STORAGE: Move dataset to SSD (current: likely HDD)")
    recommendations.append("   → Or consider caching dataset to RAM/fast storage")

# Based on num_workers
if best_workers > 0:
    recommendations.append(f"2. 👷 WORKERS: Use num_workers={best_workers} in your DataLoader")
    if device.type == 'cuda':
        recommendations.append("   → Also enable pin_memory=True for GPU training")

# Based on random/sequential ratio
if ratio > 2.0:
    recommendations.append("3. 🔀 DISK SEEKS: High random access penalty detected")
    recommendations.append("   → Consider preprocessing images to a single file format (HDF5/LMDB)")

# General recommendations
recommendations.append("4. 🎯 BATCH SIZE: If GPU memory allows, increase batch_size (32→64→128)")
recommendations.append("5. 🔄 PERSISTENT WORKERS: Use persistent_workers=True (already tested above)")

if device.type == 'cpu':
    recommendations.append("6. 🖥️  GPU: Consider using GPU for training (currently on CPU)")

if len(recommendations) == 0:
    print("\n✅ Your data loading is already well optimized!")
else:
    for rec in recommendations:
        print(f"\n{rec}")

# ==============================================================================
# Suggested DataLoader Configuration
# ==============================================================================
print("\n" + "="*80)
print("🔧 SUGGESTED DATALOADER CONFIGURATION")
print("="*80)

print(f"""
loader = DataLoader(
    dataset,
    batch_size=32,  # Increase to 64 or 128 if GPU memory allows
    num_workers={best_workers if valid_results else 4},
    pin_memory={device.type == 'cuda'},
    persistent_workers=True,
    shuffle=True  # For training
)
""")

print("="*80)
print("PROFILING COMPLETE!")
print("="*80)
