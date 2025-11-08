"""
CropShield AI - Convert Dataset to WebDataset Shards
Creates .tar shards for fast sequential reads during training

Benefits of WebDataset:
- Sequential disk reads (10-100x faster than random access)
- Streaming from disk/network
- Better caching and prefetching
- Reduced filesystem overhead (fewer inodes)
- Efficient for large-scale training

Shard format: Each .tar contains:
  - <class>__<image_name>.jpg (image file)
  - <class>__<image_name>.cls (class label as text)
"""

import os
import sys
import tarfile
from pathlib import Path
from tqdm import tqdm
import json
import hashlib
from collections import defaultdict


def calculate_md5(filepath):
    """Calculate MD5 hash for file integrity verification"""
    hash_md5 = hashlib.md5()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()


def collect_all_images(data_dir):
    """
    Collect all images with their class labels
    
    Returns:
        list: [(image_path, class_name, class_idx), ...]
    """
    data_dir = Path(data_dir)
    
    # Get all class directories
    class_dirs = sorted([d for d in data_dir.iterdir() if d.is_dir()])
    classes = [d.name for d in class_dirs]
    class_to_idx = {cls_name: idx for idx, cls_name in enumerate(classes)}
    
    print(f"📁 Found {len(classes)} classes:")
    for cls_name in classes:
        print(f"   - {cls_name}")
    
    # Collect all images
    all_samples = []
    valid_extensions = {'.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG'}
    
    print("\n🔍 Scanning for images...")
    for class_dir in tqdm(class_dirs, desc="Classes"):
        class_name = class_dir.name
        class_idx = class_to_idx[class_name]
        
        for img_path in class_dir.iterdir():
            if img_path.suffix in valid_extensions:
                all_samples.append((img_path, class_name, class_idx))
    
    print(f"\n✅ Found {len(all_samples)} total images")
    
    return all_samples, classes, class_to_idx


def create_shards(samples, output_dir, samples_per_shard=5000, split='train'):
    """
    Create WebDataset .tar shards
    
    Args:
        samples: List of (image_path, class_name, class_idx)
        output_dir: Directory to save shards
        samples_per_shard: Number of images per shard
        split: 'train', 'val', or 'test'
    
    Returns:
        dict: Shard metadata
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    num_shards = (len(samples) + samples_per_shard - 1) // samples_per_shard
    
    print(f"\n📦 Creating {num_shards} shards ({samples_per_shard} images each)...")
    
    shard_metadata = {
        'split': split,
        'total_samples': len(samples),
        'num_shards': num_shards,
        'samples_per_shard': samples_per_shard,
        'shards': []
    }
    
    # Create shards
    for shard_idx in range(num_shards):
        start_idx = shard_idx * samples_per_shard
        end_idx = min(start_idx + samples_per_shard, len(samples))
        shard_samples = samples[start_idx:end_idx]
        
        # Shard naming: <split>-<shard_number>-<total_shards>.tar
        shard_name = f"{split}-{shard_idx:06d}-{num_shards:06d}.tar"
        shard_path = output_dir / shard_name
        
        # Create tar file
        with tarfile.open(shard_path, 'w') as tar:
            for sample_idx, (img_path, class_name, class_idx) in enumerate(tqdm(
                shard_samples, 
                desc=f"Shard {shard_idx+1}/{num_shards}",
                leave=False
            )):
                # Generate unique sample ID within shard
                sample_id = f"{shard_idx:06d}_{sample_idx:06d}"
                
                # Naming format: <class>__<original_name>
                # This preserves class info in filename for easy debugging
                original_name = img_path.stem
                base_name = f"{class_name}__{original_name}"
                
                # Add image file
                img_arcname = f"{sample_id}.jpg"
                tar.add(img_path, arcname=img_arcname)
                
                # Add class label file (.cls contains class index)
                cls_arcname = f"{sample_id}.cls"
                cls_info = tarfile.TarInfo(name=cls_arcname)
                cls_data = str(class_idx).encode('utf-8')
                cls_info.size = len(cls_data)
                tar.addfile(cls_info, fileobj=__import__('io').BytesIO(cls_data))
                
                # Add class name file (.txt contains human-readable class name)
                txt_arcname = f"{sample_id}.txt"
                txt_info = tarfile.TarInfo(name=txt_arcname)
                txt_data = class_name.encode('utf-8')
                txt_info.size = len(txt_data)
                tar.addfile(txt_info, fileobj=__import__('io').BytesIO(txt_data))
        
        # Calculate shard metadata
        shard_size = shard_path.stat().st_size
        shard_md5 = calculate_md5(shard_path)
        
        shard_info = {
            'name': shard_name,
            'num_samples': len(shard_samples),
            'size_mb': shard_size / (1024 * 1024),
            'md5': shard_md5
        }
        shard_metadata['shards'].append(shard_info)
        
        print(f"   ✅ {shard_name}: {len(shard_samples)} samples, {shard_size/(1024*1024):.1f} MB")
    
    return shard_metadata


def save_metadata(metadata, output_dir, classes, class_to_idx):
    """Save shard metadata and class information"""
    output_dir = Path(output_dir)
    
    # Save full metadata
    metadata_path = output_dir / f"{metadata['split']}_metadata.json"
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"\n💾 Metadata saved to: {metadata_path}")
    
    # Save class information
    class_info = {
        'classes': classes,
        'class_to_idx': class_to_idx,
        'num_classes': len(classes)
    }
    class_info_path = output_dir / 'class_info.json'
    with open(class_info_path, 'w') as f:
        json.dump(class_info, f, indent=2)
    print(f"💾 Class info saved to: {class_info_path}")


def split_dataset(samples, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1):
    """
    Split dataset into train/val/test maintaining class distribution
    
    Returns:
        tuple: (train_samples, val_samples, test_samples)
    """
    from collections import defaultdict
    import random
    
    # Group by class
    class_samples = defaultdict(list)
    for sample in samples:
        class_name = sample[1]
        class_samples[class_name].append(sample)
    
    train_samples = []
    val_samples = []
    test_samples = []
    
    print("\n🔀 Splitting dataset (stratified by class)...")
    for class_name, class_imgs in class_samples.items():
        # Shuffle within class
        random.shuffle(class_imgs)
        
        n = len(class_imgs)
        n_train = int(n * train_ratio)
        n_val = int(n * val_ratio)
        
        train_samples.extend(class_imgs[:n_train])
        val_samples.extend(class_imgs[n_train:n_train + n_val])
        test_samples.extend(class_imgs[n_train + n_val:])
    
    # Shuffle final splits
    random.shuffle(train_samples)
    random.shuffle(val_samples)
    random.shuffle(test_samples)
    
    print(f"   Train: {len(train_samples)} samples ({len(train_samples)/len(samples)*100:.1f}%)")
    print(f"   Val:   {len(val_samples)} samples ({len(val_samples)/len(samples)*100:.1f}%)")
    print(f"   Test:  {len(test_samples)} samples ({len(test_samples)/len(samples)*100:.1f}%)")
    
    return train_samples, val_samples, test_samples


def main():
    """Main conversion pipeline"""
    
    print("=" * 80)
    print("CROPSHIELD AI - WEBDATASET SHARD CREATION")
    print("=" * 80)
    
    # Configuration
    DATA_DIR = "Database_resized/"
    OUTPUT_DIR = "shards/"
    SAMPLES_PER_SHARD = 5000
    
    # Split ratios
    TRAIN_RATIO = 0.8
    VAL_RATIO = 0.1
    TEST_RATIO = 0.1
    
    # Set random seed for reproducibility
    import random
    random.seed(42)
    
    print(f"\n⚙️  CONFIGURATION:")
    print(f"   Input:  {DATA_DIR}")
    print(f"   Output: {OUTPUT_DIR}")
    print(f"   Samples per shard: {SAMPLES_PER_SHARD}")
    print(f"   Split: {TRAIN_RATIO:.0%} train / {VAL_RATIO:.0%} val / {TEST_RATIO:.0%} test")
    
    # Collect all images
    all_samples, classes, class_to_idx = collect_all_images(DATA_DIR)
    
    # Split dataset
    train_samples, val_samples, test_samples = split_dataset(
        all_samples, 
        TRAIN_RATIO, 
        VAL_RATIO, 
        TEST_RATIO
    )
    
    # Create output directory
    output_dir = Path(OUTPUT_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create shards for each split
    all_metadata = {}
    
    for split_name, split_samples in [
        ('train', train_samples),
        ('val', val_samples),
        ('test', test_samples)
    ]:
        if len(split_samples) > 0:
            print(f"\n{'='*80}")
            print(f"CREATING {split_name.upper()} SHARDS")
            print(f"{'='*80}")
            
            metadata = create_shards(
                split_samples, 
                output_dir, 
                SAMPLES_PER_SHARD, 
                split_name
            )
            save_metadata(metadata, output_dir, classes, class_to_idx)
            all_metadata[split_name] = metadata
    
    # Print summary
    print("\n" + "="*80)
    print("CONVERSION COMPLETE!")
    print("="*80)
    
    total_size = 0
    for split_name, metadata in all_metadata.items():
        split_size = sum(s['size_mb'] for s in metadata['shards'])
        total_size += split_size
        print(f"\n📊 {split_name.upper()}:")
        print(f"   Samples: {metadata['total_samples']}")
        print(f"   Shards:  {metadata['num_shards']}")
        print(f"   Size:    {split_size:.1f} MB")
    
    print(f"\n💾 Total size: {total_size:.1f} MB")
    print(f"📁 Output directory: {OUTPUT_DIR}")
    
    print("\n✅ Next steps:")
    print("   1. Verify shards: python scripts/verify_webdataset_shards.py")
    print("   2. Test loading: python webdataset_loader.py")
    print("   3. Start training with WebDataset!")


if __name__ == "__main__":
    main()
