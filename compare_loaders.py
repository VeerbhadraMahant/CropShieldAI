"""
CropShield AI - WebDataset vs Standard DataLoader Comparison
Compare performance of different data loading strategies

Compares:
1. Original PIL + Random Access (Database_resized/)
2. FastImageFolder + torchvision.io (Database_resized/)
3. WebDataset + Sequential Reads (shards/)
"""

import torch
import time
from pathlib import Path
import sys


def benchmark_loader(loader, num_batches=50, name="Loader"):
    """Benchmark a DataLoader"""
    print(f"\n🧪 Testing {name}...")
    
    # Warmup
    for i, (images, labels) in enumerate(loader):
        if i >= 5:
            break
    
    # Actual benchmark
    start_time = time.time()
    batch_count = 0
    total_images = 0
    
    for images, labels in loader:
        batch_count += 1
        total_images += images.size(0)
        
        if batch_count >= num_batches:
            break
    
    elapsed = time.time() - start_time
    throughput = total_images / elapsed
    
    print(f"   Batches: {batch_count}")
    print(f"   Images: {total_images}")
    print(f"   Time: {elapsed:.2f}s")
    print(f"   Throughput: {throughput:.1f} img/s")
    
    return {
        'name': name,
        'batches': batch_count,
        'images': total_images,
        'time': elapsed,
        'throughput': throughput
    }


def test_fastdataset():
    """Test FastImageFolder loader"""
    from fast_dataset import make_loaders
    
    if not Path("Database_resized/").exists():
        print("❌ Database_resized/ not found, skipping...")
        return None
    
    train_loader, _, _, _, _ = make_loaders(
        data_dir='Database_resized/',
        batch_size=32,
        num_workers=12
    )
    
    return benchmark_loader(train_loader, num_batches=50, name="FastImageFolder")


def test_webdataset():
    """Test WebDataset loader"""
    from webdataset_loader import make_webdataset_loaders
    
    if not Path("shards/").exists():
        print("❌ shards/ not found, skipping...")
        return None
    
    train_loader, _, _, _ = make_webdataset_loaders(
        shards_dir='shards/',
        batch_size=32,
        num_workers=12
    )
    
    return benchmark_loader(train_loader, num_batches=50, name="WebDataset")


def print_comparison(results):
    """Print comparison table"""
    print("\n" + "="*80)
    print("PERFORMANCE COMPARISON")
    print("="*80)
    
    valid_results = [r for r in results if r is not None]
    
    if not valid_results:
        print("❌ No valid results to compare")
        return
    
    # Header
    print(f"\n{'Method':<30} {'Throughput':<15} {'Time':<10} {'Speedup':<10}")
    print("-" * 80)
    
    # Baseline (first result)
    baseline_throughput = valid_results[0]['throughput']
    
    # Print results
    for result in valid_results:
        speedup = result['throughput'] / baseline_throughput
        
        print(f"{result['name']:<30} "
              f"{result['throughput']:>8.1f} img/s   "
              f"{result['time']:>6.2f}s   "
              f"{speedup:>6.2f}x")
    
    # Best configuration
    best = max(valid_results, key=lambda x: x['throughput'])
    print("\n" + "="*80)
    print(f"🏆 BEST: {best['name']}")
    print(f"   Throughput: {best['throughput']:.1f} img/s")
    print(f"   Speedup: {best['throughput']/baseline_throughput:.2f}x over {valid_results[0]['name']}")
    print("="*80)


def main():
    print("="*80)
    print("CROPSHIELD AI - DATA LOADING COMPARISON")
    print("="*80)
    
    print("\nComparing data loading strategies:")
    print("1. FastImageFolder (torchvision.io + preprocessed)")
    print("2. WebDataset (sequential tar shards)")
    
    results = []
    
    # Test FastImageFolder
    print("\n" + "="*80)
    print("TEST 1: FastImageFolder")
    print("="*80)
    result = test_fastdataset()
    if result:
        results.append(result)
    
    # Test WebDataset
    print("\n" + "="*80)
    print("TEST 2: WebDataset")
    print("="*80)
    result = test_webdataset()
    if result:
        results.append(result)
    
    # Print comparison
    print_comparison(results)
    
    # Recommendations
    print("\n💡 RECOMMENDATIONS:")
    print("   - Use WebDataset for maximum throughput (sequential reads)")
    print("   - Especially beneficial on HDD (10-100x faster than random access)")
    print("   - Enables efficient multi-worker loading without file system bottlenecks")
    print("   - Better OS-level caching and prefetching")
    
    print("\n📚 Next steps:")
    print("   1. Use WebDataset in training: from webdataset_loader import make_webdataset_loaders")
    print("   2. Build CNN model architecture")
    print("   3. Implement training loop with optimal configuration")


if __name__ == "__main__":
    main()
