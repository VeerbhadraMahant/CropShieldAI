"""
CropShield AI - Detailed Component Profiler
Tests individual operations: disk read, decode, resize, transform
"""

import time
import os
from PIL import Image
import torchvision.transforms as T
import glob

print("="*80)
print("CROPSHIELD AI - DETAILED COMPONENT PROFILER")
print("="*80)

# Find sample images from different classes
dataset_path = 'Database/'
sample_images = []

print(f"\n📁 Scanning for sample images...")
for class_folder in os.listdir(dataset_path):
    class_path = os.path.join(dataset_path, class_folder)
    if os.path.isdir(class_path):
        images = glob.glob(os.path.join(class_path, '*.[jJ][pP][gG]')) + \
                 glob.glob(os.path.join(class_path, '*.[jJ][pP][eE][gG]')) + \
                 glob.glob(os.path.join(class_path, '*.[pP][nN][gG]'))
        if images:
            sample_images.append(images[0])
            if len(sample_images) >= 5:
                break

if not sample_images:
    print("❌ Error: No images found in Database folder!")
    exit(1)

print(f"✅ Found {len(sample_images)} sample images")

# Test each component for multiple images
print("\n" + "="*80)
print("TESTING INDIVIDUAL COMPONENTS (averaged over 5 samples)")
print("="*80)

disk_times = []
decode_times = []
resize_times = []
tensor_times = []
total_times = []

for img_path in sample_images:
    # 1. Disk Read
    start = time.time()
    with open(img_path, 'rb') as f:
        data = f.read()
    disk_time = time.time() - start
    disk_times.append(disk_time)
    
    # 2. Image Decode
    start = time.time()
    img = Image.open(img_path)
    img = img.convert('RGB')
    _ = img.size  # Force decode
    decode_time = time.time() - start
    decode_times.append(decode_time)
    
    # 3. Resize
    start = time.time()
    resizer = T.Resize((224, 224))
    img_resized = resizer(img)
    resize_time = time.time() - start
    resize_times.append(resize_time)
    
    # 4. ToTensor + Normalize
    start = time.time()
    transform = T.Compose([
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    tensor = transform(img_resized)
    tensor_time = time.time() - start
    tensor_times.append(tensor_time)
    
    total_times.append(disk_time + decode_time + resize_time + tensor_time)

# Calculate averages
avg_disk = sum(disk_times) / len(disk_times)
avg_decode = sum(decode_times) / len(decode_times)
avg_resize = sum(resize_times) / len(resize_times)
avg_tensor = sum(tensor_times) / len(tensor_times)
avg_total = sum(total_times) / len(total_times)

print(f"\n📊 COMPONENT BREAKDOWN (average):")
print(f"   1️⃣  Disk Read:        {avg_disk*1000:6.2f}ms ({avg_disk/avg_total*100:5.1f}%)")
print(f"   2️⃣  Image Decode:     {avg_decode*1000:6.2f}ms ({avg_decode/avg_total*100:5.1f}%)")
print(f"   3️⃣  Resize:           {avg_resize*1000:6.2f}ms ({avg_resize/avg_total*100:5.1f}%)")
print(f"   4️⃣  ToTensor+Norm:    {avg_tensor*1000:6.2f}ms ({avg_tensor/avg_total*100:5.1f}%)")
print(f"   " + "-"*40)
print(f"   📦 TOTAL PER IMAGE:  {avg_total*1000:6.2f}ms")

# Identify bottleneck
components = {
    'Disk Read': avg_disk,
    'Image Decode': avg_decode,
    'Resize': avg_resize,
    'ToTensor+Normalize': avg_tensor
}
bottleneck = max(components.items(), key=lambda x: x[1])

print(f"\n🎯 PRIMARY BOTTLENECK: {bottleneck[0]} ({bottleneck[1]/avg_total*100:.1f}% of total time)")

# Recommendations based on bottleneck
print("\n" + "="*80)
print("💡 COMPONENT-SPECIFIC RECOMMENDATIONS")
print("="*80)

if bottleneck[0] == 'Disk Read':
    print("\n❌ DISK READ is the bottleneck")
    print("   Solutions:")
    print("   → Move dataset to SSD/NVMe storage")
    print("   → Increase DataLoader num_workers (parallel I/O)")
    print("   → Consider caching to RAM if dataset fits")
    print("   → Use faster file system (ext4/NTFS)")
    
elif bottleneck[0] == 'Image Decode':
    print("\n❌ IMAGE DECODE is the bottleneck")
    print("   Solutions:")
    print("   → Use JPEG with lower quality (faster decode)")
    print("   → Convert images to PNG (if applicable)")
    print("   → Increase num_workers (parallel decoding)")
    print("   → Consider preprocessing images to tensors")
    
elif bottleneck[0] == 'Resize':
    print("\n❌ RESIZE is the bottleneck")
    print("   Solutions:")
    print("   → Pre-resize all images to 224x224 offline")
    print("   → Use faster interpolation (BILINEAR vs BICUBIC)")
    print("   → Increase num_workers (parallel processing)")
    
elif bottleneck[0] == 'ToTensor+Normalize':
    print("\n❌ TENSOR CONVERSION is the bottleneck")
    print("   Solutions:")
    print("   → This is usually fast, but increase num_workers")
    print("   → Consider GPU-based augmentation libraries")

# Overall assessment
print("\n" + "="*80)
print("📈 PERFORMANCE ASSESSMENT")
print("="*80)

if avg_total < 0.03:
    print("\n✅ EXCELLENT: Very fast per-image processing (<30ms)")
    print("   → Your pipeline is well optimized")
elif avg_total < 0.05:
    print("\n✅ GOOD: Acceptable per-image processing (30-50ms)")
    print("   → Should achieve good training throughput")
elif avg_total < 0.1:
    print("\n⚠️  MODERATE: Slow per-image processing (50-100ms)")
    print("   → Consider optimization strategies above")
else:
    print("\n❌ SLOW: Very slow per-image processing (>100ms)")
    print("   → Significant optimization needed")

# Calculate throughput
print(f"\n📊 ESTIMATED THROUGHPUT:")
print(f"   Single-threaded: {1/avg_total:.1f} images/second")
print(f"   With 4 workers:  {4/avg_total:.1f} images/second (theoretical)")
print(f"   With 8 workers:  {8/avg_total:.1f} images/second (theoretical)")

print("\n" + "="*80)
print("COMPONENT PROFILING COMPLETE!")
print("="*80)
