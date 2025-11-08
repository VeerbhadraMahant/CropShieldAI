"""
CropShield AI - Image Preprocessing Script
Resize all images to 224×224 for faster training

This script preprocesses the entire dataset once, eliminating the need
for runtime resizing during training. Uses multiprocessing for speed.

Usage:
    python scripts/resize_images.py

Requirements:
    pip install pillow-simd  # Faster than regular Pillow
"""

import os
import time
import shutil
from pathlib import Path
from PIL import Image
from multiprocessing import Pool, cpu_count
from tqdm import tqdm
import glob

# Configuration
SOURCE_DIR = 'Database'
TARGET_DIR = 'Database_resized'
TARGET_SIZE = (224, 224)
IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'}
NUM_WORKERS = max(1, cpu_count() - 1)  # Leave 1 CPU core free

# Statistics
stats = {
    'total': 0,
    'processed': 0,
    'skipped': 0,
    'failed': 0,
    'start_time': None,
    'end_time': None
}


def is_image_file(filepath):
    """Check if file is a valid image"""
    return filepath.suffix.lower() in IMAGE_EXTENSIONS


def get_all_image_paths(source_dir):
    """
    Recursively find all images in source directory
    Returns list of (source_path, class_name, filename) tuples
    """
    image_paths = []
    source_path = Path(source_dir)
    
    if not source_path.exists():
        raise FileNotFoundError(f"Source directory '{source_dir}' not found!")
    
    # Iterate through class folders
    for class_dir in sorted(source_path.iterdir()):
        if not class_dir.is_dir():
            continue
        
        class_name = class_dir.name
        
        # Find all images in this class folder
        for ext in IMAGE_EXTENSIONS:
            # Case-insensitive search
            for pattern in [f'*{ext}', f'*{ext.upper()}']:
                for img_path in class_dir.glob(pattern):
                    if img_path.is_file():
                        image_paths.append((img_path, class_name, img_path.name))
    
    return image_paths


def process_single_image(args):
    """
    Process a single image: resize and save
    
    Args:
        args: Tuple of (source_path, class_name, filename, target_dir, target_size)
    
    Returns:
        Tuple of (status, message) where status is 'success', 'skipped', or 'failed'
    """
    source_path, class_name, filename, target_dir, target_size = args
    
    # Create target directory if needed
    target_class_dir = Path(target_dir) / class_name
    target_class_dir.mkdir(parents=True, exist_ok=True)
    
    # Target file path
    target_path = target_class_dir / filename
    
    # Skip if already processed
    if target_path.exists():
        return ('skipped', f"Already exists: {target_path}")
    
    try:
        # Open image
        with Image.open(source_path) as img:
            # Convert to RGB (handles RGBA, grayscale, etc.)
            if img.mode != 'RGB':
                img = img.convert('RGB')
            
            # Resize using high-quality Lanczos filter
            img_resized = img.resize(target_size, Image.LANCZOS)
            
            # Save with good quality
            img_resized.save(target_path, quality=95, optimize=True)
        
        return ('success', f"Processed: {source_path} -> {target_path}")
    
    except Exception as e:
        return ('failed', f"Failed: {source_path} | Error: {str(e)}")


def process_images_parallel(image_paths, target_dir, target_size, num_workers):
    """
    Process all images using multiprocessing
    
    Args:
        image_paths: List of (source_path, class_name, filename) tuples
        target_dir: Target directory path
        target_size: Tuple of (width, height)
        num_workers: Number of worker processes
    """
    # Prepare arguments for each image
    args_list = [
        (source_path, class_name, filename, target_dir, target_size)
        for source_path, class_name, filename in image_paths
    ]
    
    print(f"\n🔄 Processing {len(image_paths)} images with {num_workers} workers...")
    print(f"   Source: {SOURCE_DIR}/")
    print(f"   Target: {TARGET_DIR}/")
    print(f"   Size: {target_size[0]}×{target_size[1]}")
    
    # Process with progress bar
    with Pool(num_workers) as pool:
        results = list(tqdm(
            pool.imap(process_single_image, args_list),
            total=len(args_list),
            desc="Resizing images",
            unit="img"
        ))
    
    # Count results
    for status, message in results:
        if status == 'success':
            stats['processed'] += 1
        elif status == 'skipped':
            stats['skipped'] += 1
        elif status == 'failed':
            stats['failed'] += 1
            print(f"   ⚠️  {message}")
    
    return results


def print_statistics():
    """Print processing statistics"""
    elapsed_time = stats['end_time'] - stats['start_time']
    
    print("\n" + "="*80)
    print("PREPROCESSING COMPLETE!")
    print("="*80)
    
    print(f"\n📊 STATISTICS:")
    print(f"   Total images found:  {stats['total']:,}")
    print(f"   ✅ Successfully processed: {stats['processed']:,}")
    print(f"   ⏭️  Skipped (already exist): {stats['skipped']:,}")
    print(f"   ❌ Failed: {stats['failed']:,}")
    
    print(f"\n⏱️  PERFORMANCE:")
    print(f"   Total time: {elapsed_time:.2f} seconds")
    
    if stats['processed'] > 0:
        images_per_sec = stats['processed'] / elapsed_time
        time_per_image = elapsed_time / stats['processed'] * 1000
        print(f"   Throughput: {images_per_sec:.1f} images/second")
        print(f"   Time per image: {time_per_image:.1f}ms")
    
    # Calculate space savings
    if stats['processed'] > 0:
        print(f"\n💾 STORAGE:")
        source_size = get_dir_size(SOURCE_DIR)
        target_size = get_dir_size(TARGET_DIR)
        print(f"   Original dataset: {source_size / (1024**3):.2f} GB")
        print(f"   Resized dataset: {target_size / (1024**3):.2f} GB")
        if source_size > 0:
            savings = (1 - target_size / source_size) * 100
            print(f"   Space saved: {savings:.1f}%")
    
    print(f"\n📁 OUTPUT DIRECTORY:")
    print(f"   {Path(TARGET_DIR).absolute()}")
    
    print("\n✅ Ready for training! Use 'Database_resized/' as your dataset path.")
    print("="*80)


def get_dir_size(path):
    """Calculate total size of directory in bytes"""
    total_size = 0
    try:
        for dirpath, dirnames, filenames in os.walk(path):
            for filename in filenames:
                filepath = os.path.join(dirpath, filename)
                if os.path.exists(filepath):
                    total_size += os.path.getsize(filepath)
    except Exception:
        pass
    return total_size


def verify_output():
    """Verify the output directory structure"""
    target_path = Path(TARGET_DIR)
    
    if not target_path.exists():
        print(f"⚠️  Warning: Target directory '{TARGET_DIR}' does not exist yet")
        return
    
    # Count classes and images
    class_counts = {}
    for class_dir in sorted(target_path.iterdir()):
        if class_dir.is_dir():
            image_count = len(list(class_dir.glob('*.*')))
            class_counts[class_dir.name] = image_count
    
    print(f"\n📊 OUTPUT STRUCTURE:")
    print(f"   Total classes: {len(class_counts)}")
    print(f"   Class distribution:")
    
    for class_name, count in sorted(class_counts.items())[:10]:
        print(f"      {class_name}: {count:,} images")
    
    if len(class_counts) > 10:
        print(f"      ... and {len(class_counts) - 10} more classes")


def check_pillow_simd():
    """Check if Pillow-SIMD is installed"""
    try:
        import PIL
        if hasattr(PIL, 'PILLOW_VERSION'):
            version = PIL.PILLOW_VERSION
        else:
            version = PIL.__version__
        
        # Check if SIMD optimizations are available
        if 'pillow-simd' in version.lower() or 'simd' in str(PIL.__dict__):
            print("✅ Pillow-SIMD detected (optimized)")
            return True
        else:
            print("⚠️  Using standard Pillow (not SIMD-optimized)")
            print("   For 2-4x faster processing, install: pip install pillow-simd")
            return False
    except Exception:
        print("⚠️  Could not detect Pillow version")
        return False


def main():
    """Main preprocessing pipeline"""
    print("="*80)
    print("CROPSHIELD AI - IMAGE PREPROCESSING")
    print("="*80)
    
    # Check Pillow version
    print("\n🔍 Checking dependencies...")
    check_pillow_simd()
    
    # Verify source directory exists
    if not os.path.exists(SOURCE_DIR):
        print(f"\n❌ Error: Source directory '{SOURCE_DIR}' not found!")
        print(f"   Current directory: {os.getcwd()}")
        return
    
    # Get all image paths
    print(f"\n📁 Scanning source directory: {SOURCE_DIR}/")
    try:
        image_paths = get_all_image_paths(SOURCE_DIR)
        stats['total'] = len(image_paths)
        print(f"✅ Found {stats['total']:,} images")
        
        if stats['total'] == 0:
            print("\n⚠️  No images found! Check your source directory.")
            return
        
        # Count images per class
        class_counts = {}
        for _, class_name, _ in image_paths:
            class_counts[class_name] = class_counts.get(class_name, 0) + 1
        
        print(f"✅ Found {len(class_counts)} classes:")
        for class_name, count in sorted(class_counts.items())[:10]:
            print(f"   {class_name}: {count:,} images")
        if len(class_counts) > 10:
            print(f"   ... and {len(class_counts) - 10} more classes")
        
    except Exception as e:
        print(f"\n❌ Error scanning directory: {e}")
        return
    
    # Confirm processing
    print(f"\n⚙️  Configuration:")
    print(f"   Source: {SOURCE_DIR}/")
    print(f"   Target: {TARGET_DIR}/")
    print(f"   Target size: {TARGET_SIZE[0]}×{TARGET_SIZE[1]}")
    print(f"   Workers: {NUM_WORKERS}")
    print(f"   Images to process: {stats['total']:,}")
    
    # Check if target directory exists
    if os.path.exists(TARGET_DIR):
        existing_count = sum(1 for _ in Path(TARGET_DIR).rglob('*.*'))
        if existing_count > 0:
            print(f"\n⚠️  Target directory exists with {existing_count:,} files")
            print("   Existing files will be skipped (no reprocessing)")
    
    response = input("\n🚀 Start preprocessing? [Y/n]: ").strip().lower()
    if response and response not in ['y', 'yes']:
        print("❌ Preprocessing cancelled.")
        return
    
    # Start processing
    stats['start_time'] = time.time()
    
    # Create target directory
    Path(TARGET_DIR).mkdir(parents=True, exist_ok=True)
    
    # Process all images
    results = process_images_parallel(
        image_paths,
        TARGET_DIR,
        TARGET_SIZE,
        NUM_WORKERS
    )
    
    stats['end_time'] = time.time()
    
    # Verify output
    verify_output()
    
    # Print statistics
    print_statistics()
    
    # Save processing log
    log_file = Path(TARGET_DIR) / 'preprocessing_log.txt'
    with open(log_file, 'w') as f:
        f.write("CropShield AI - Preprocessing Log\n")
        f.write("="*80 + "\n\n")
        f.write(f"Date: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Source: {SOURCE_DIR}/\n")
        f.write(f"Target: {TARGET_DIR}/\n")
        f.write(f"Target size: {TARGET_SIZE[0]}×{TARGET_SIZE[1]}\n")
        f.write(f"Workers: {NUM_WORKERS}\n\n")
        f.write(f"Total images: {stats['total']:,}\n")
        f.write(f"Processed: {stats['processed']:,}\n")
        f.write(f"Skipped: {stats['skipped']:,}\n")
        f.write(f"Failed: {stats['failed']:,}\n")
        f.write(f"Time: {stats['end_time'] - stats['start_time']:.2f}s\n")
    
    print(f"\n📝 Log saved to: {log_file}")


if __name__ == '__main__':
    """
    Entry point for script
    This is required for Windows multiprocessing
    """
    main()
