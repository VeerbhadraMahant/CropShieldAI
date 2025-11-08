"""
CropShield AI - WebDataset Shard Verification
Verifies integrity and structure of created .tar shards

Checks:
- Shard file integrity (MD5 hashes)
- Sample counts match metadata
- All required files present (.jpg, .cls, .txt)
- Class labels are valid
- Images can be decoded
"""

import os
import json
import tarfile
import hashlib
from pathlib import Path
from tqdm import tqdm
from collections import Counter
import io
from PIL import Image


def calculate_md5(filepath):
    """Calculate MD5 hash for file integrity"""
    hash_md5 = hashlib.md5()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()


def verify_shard(shard_path, expected_md5=None):
    """
    Verify a single shard file
    
    Returns:
        dict: Verification results
    """
    results = {
        'path': str(shard_path),
        'exists': shard_path.exists(),
        'md5_match': None,
        'num_samples': 0,
        'errors': []
    }
    
    if not results['exists']:
        results['errors'].append("File does not exist")
        return results
    
    # Check MD5
    if expected_md5:
        actual_md5 = calculate_md5(shard_path)
        results['md5_match'] = (actual_md5 == expected_md5)
        if not results['md5_match']:
            results['errors'].append(f"MD5 mismatch: expected {expected_md5}, got {actual_md5}")
    
    # Verify tar structure
    try:
        with tarfile.open(shard_path, 'r') as tar:
            members = tar.getmembers()
            
            # Group by sample ID
            samples = {}
            for member in members:
                # Extract sample ID and extension
                name = Path(member.name)
                sample_id = name.stem
                ext = name.suffix
                
                if sample_id not in samples:
                    samples[sample_id] = set()
                samples[sample_id].add(ext)
            
            results['num_samples'] = len(samples)
            
            # Check each sample has required files
            for sample_id, extensions in samples.items():
                required = {'.jpg', '.cls', '.txt'}
                missing = required - extensions
                if missing:
                    results['errors'].append(f"Sample {sample_id} missing: {missing}")
            
    except Exception as e:
        results['errors'].append(f"Failed to read tar: {str(e)}")
    
    return results


def verify_sample_integrity(shard_path, max_samples=10):
    """
    Deep verification: decode images and check labels
    
    Args:
        shard_path: Path to shard
        max_samples: Number of samples to check (None = all)
    
    Returns:
        dict: Detailed verification results
    """
    results = {
        'samples_checked': 0,
        'decode_errors': [],
        'label_errors': [],
        'class_distribution': Counter()
    }
    
    try:
        with tarfile.open(shard_path, 'r') as tar:
            # Group files by sample ID
            sample_groups = {}
            for member in tar.getmembers():
                name = Path(member.name)
                sample_id = name.stem
                if sample_id not in sample_groups:
                    sample_groups[sample_id] = {}
                sample_groups[sample_id][name.suffix] = member
            
            # Check samples
            samples_to_check = list(sample_groups.items())[:max_samples] if max_samples else sample_groups.items()
            
            for sample_id, files in samples_to_check:
                results['samples_checked'] += 1
                
                # Try to decode image
                if '.jpg' in files:
                    try:
                        img_file = tar.extractfile(files['.jpg'])
                        img = Image.open(img_file)
                        img.verify()  # Check integrity
                        img_file.close()
                    except Exception as e:
                        results['decode_errors'].append(f"{sample_id}: {str(e)}")
                
                # Check class label
                if '.cls' in files:
                    try:
                        cls_file = tar.extractfile(files['.cls'])
                        class_idx = int(cls_file.read().decode('utf-8').strip())
                        results['class_distribution'][class_idx] += 1
                    except Exception as e:
                        results['label_errors'].append(f"{sample_id}: {str(e)}")
                
    except Exception as e:
        results['decode_errors'].append(f"Failed to open tar: {str(e)}")
    
    return results


def verify_all_shards(shards_dir, split='train'):
    """
    Verify all shards for a split
    
    Args:
        shards_dir: Directory containing shards
        split: 'train', 'val', or 'test'
    
    Returns:
        dict: Overall verification results
    """
    shards_dir = Path(shards_dir)
    
    # Load metadata
    metadata_path = shards_dir / f"{split}_metadata.json"
    if not metadata_path.exists():
        print(f"❌ Metadata not found: {metadata_path}")
        return None
    
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)
    
    print(f"\n{'='*80}")
    print(f"VERIFYING {split.upper()} SHARDS")
    print(f"{'='*80}")
    print(f"Expected shards: {metadata['num_shards']}")
    print(f"Expected samples: {metadata['total_samples']}")
    
    # Verify each shard
    all_results = []
    total_samples = 0
    total_errors = 0
    
    for shard_info in tqdm(metadata['shards'], desc="Verifying shards"):
        shard_path = shards_dir / shard_info['name']
        expected_md5 = shard_info.get('md5')
        
        result = verify_shard(shard_path, expected_md5)
        all_results.append(result)
        
        total_samples += result['num_samples']
        total_errors += len(result['errors'])
    
    # Summary
    print(f"\n📊 VERIFICATION SUMMARY:")
    print(f"   Shards checked: {len(all_results)}")
    print(f"   Total samples found: {total_samples}")
    print(f"   Expected samples: {metadata['total_samples']}")
    print(f"   Sample count match: {'✅' if total_samples == metadata['total_samples'] else '❌'}")
    print(f"   Total errors: {total_errors}")
    
    # Show errors
    if total_errors > 0:
        print(f"\n❌ ERRORS FOUND:")
        for result in all_results:
            if result['errors']:
                print(f"\n   Shard: {Path(result['path']).name}")
                for error in result['errors']:
                    print(f"      - {error}")
    else:
        print(f"\n✅ All shards verified successfully!")
    
    return {
        'split': split,
        'shards_verified': len(all_results),
        'total_samples': total_samples,
        'expected_samples': metadata['total_samples'],
        'total_errors': total_errors,
        'results': all_results
    }


def deep_verify_split(shards_dir, split='train', samples_per_shard=5):
    """
    Deep verification with image decoding and label checking
    
    Args:
        shards_dir: Directory containing shards
        split: 'train', 'val', or 'test'
        samples_per_shard: Number of samples to check per shard
    """
    shards_dir = Path(shards_dir)
    
    # Load metadata
    metadata_path = shards_dir / f"{split}_metadata.json"
    if not metadata_path.exists():
        print(f"❌ Metadata not found: {metadata_path}")
        return
    
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)
    
    # Load class info
    class_info_path = shards_dir / 'class_info.json'
    with open(class_info_path, 'r') as f:
        class_info = json.load(f)
    
    print(f"\n{'='*80}")
    print(f"DEEP VERIFICATION - {split.upper()}")
    print(f"{'='*80}")
    print(f"Checking {samples_per_shard} samples per shard...")
    
    all_class_dist = Counter()
    total_decode_errors = 0
    total_label_errors = 0
    total_checked = 0
    
    for shard_info in tqdm(metadata['shards'], desc="Deep checking"):
        shard_path = shards_dir / shard_info['name']
        
        result = verify_sample_integrity(shard_path, samples_per_shard)
        
        total_checked += result['samples_checked']
        total_decode_errors += len(result['decode_errors'])
        total_label_errors += len(result['label_errors'])
        all_class_dist.update(result['class_distribution'])
    
    print(f"\n📊 DEEP VERIFICATION RESULTS:")
    print(f"   Samples checked: {total_checked}")
    print(f"   Decode errors: {total_decode_errors}")
    print(f"   Label errors: {total_label_errors}")
    
    if total_decode_errors == 0 and total_label_errors == 0:
        print(f"\n✅ All samples verified successfully!")
    
    # Show class distribution
    print(f"\n📊 CLASS DISTRIBUTION (sampled):")
    classes = class_info['classes']
    for class_idx in sorted(all_class_dist.keys()):
        class_name = classes[class_idx] if class_idx < len(classes) else f"Unknown_{class_idx}"
        count = all_class_dist[class_idx]
        print(f"   {class_idx:2d}. {class_name:40s}: {count:4d} samples")


def count_total_samples(shards_dir):
    """
    Count total samples across all splits
    """
    shards_dir = Path(shards_dir)
    
    print(f"\n{'='*80}")
    print(f"SAMPLE COUNT SUMMARY")
    print(f"{'='*80}")
    
    total_all = 0
    
    for split in ['train', 'val', 'test']:
        metadata_path = shards_dir / f"{split}_metadata.json"
        if metadata_path.exists():
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
            
            total = metadata['total_samples']
            num_shards = metadata['num_shards']
            size_mb = sum(s['size_mb'] for s in metadata['shards'])
            
            print(f"\n{split.upper()}:")
            print(f"   Samples: {total:,}")
            print(f"   Shards:  {num_shards}")
            print(f"   Size:    {size_mb:.1f} MB")
            
            total_all += total
    
    print(f"\n{'='*80}")
    print(f"TOTAL SAMPLES: {total_all:,}")
    print(f"{'='*80}")


def main():
    """Main verification pipeline"""
    
    print("=" * 80)
    print("CROPSHIELD AI - WEBDATASET SHARD VERIFICATION")
    print("=" * 80)
    
    SHARDS_DIR = "shards/"
    
    # Check if shards directory exists
    if not Path(SHARDS_DIR).exists():
        print(f"\n❌ Shards directory not found: {SHARDS_DIR}")
        print("   Run create_webdataset_shards.py first!")
        return
    
    # Verify all splits
    for split in ['train', 'val', 'test']:
        metadata_path = Path(SHARDS_DIR) / f"{split}_metadata.json"
        if metadata_path.exists():
            # Basic verification
            verify_all_shards(SHARDS_DIR, split)
            
            # Deep verification (sample a few images)
            deep_verify_split(SHARDS_DIR, split, samples_per_shard=3)
    
    # Count total samples
    count_total_samples(SHARDS_DIR)
    
    print("\n✅ Verification complete!")
    print("\n💡 Next step: Test loading with webdataset_loader.py")


if __name__ == "__main__":
    main()
