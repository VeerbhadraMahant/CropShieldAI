"""
CropShield AI - Performance Comparison
Compares original vs preprocessed dataset loading performance
"""

import torch
import time
from data_loader_optimized import create_optimized_dataloaders
from data_loader_fast import create_fast_dataloaders

def benchmark_dataloader(loader, name, num_batches=50):
    """
    Benchmark a DataLoader
    
    Args:
        loader: DataLoader to benchmark
        name: Name for display
        num_batches: Number of batches to test
    
    Returns:
        Dictionary with timing statistics
    """
    print(f"\n🔄 Benchmarking: {name}")
    print(f"   Testing {num_batches} batches...")
    
    times = []
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    start_total = time.time()
    
    for i, (images, labels) in enumerate(loader):
        if i >= num_batches:
            break
        
        batch_start = time.time()
        
        # Simulate moving to device (common in training)
        if device.type == 'cuda':
            images = images.to(device)
            labels = labels.to(device)
        
        batch_time = time.time() - batch_start
        times.append(batch_time)
    
    total_time = time.time() - start_total
    
    stats = {
        'total_time': total_time,
        'avg_batch_time': sum(times) / len(times),
        'min_batch_time': min(times),
        'max_batch_time': max(times),
        'throughput': (num_batches * 32) / total_time  # Assuming batch_size=32
    }
    
    print(f"   ✅ Total time: {stats['total_time']:.2f}s")
    print(f"   ✅ Avg per batch: {stats['avg_batch_time']*1000:.1f}ms")
    print(f"   ✅ Throughput: {stats['throughput']:.1f} img/s")
    
    return stats


def main():
    print("="*80)
    print("CROPSHIELD AI - PERFORMANCE COMPARISON")
    print("="*80)
    
    print("\n📊 Comparing original vs preprocessed dataset loading...")
    
    # Test configuration
    batch_size = 32
    num_batches = 50
    num_workers = 0
    
    print(f"\n⚙️  Test Configuration:")
    print(f"   Batch size: {batch_size}")
    print(f"   Test batches: {num_batches}")
    print(f"   Total images: {num_batches * batch_size}")
    print(f"   num_workers: {num_workers}")
    
    # Create original dataloaders
    print("\n" + "="*80)
    print("LOADING ORIGINAL DATASET (with runtime resize)")
    print("="*80)
    
    try:
        train_orig, _, _, _, info_orig = create_optimized_dataloaders(
            data_dir='Database/',
            batch_size=batch_size,
            num_workers=num_workers
        )
        
        stats_orig = benchmark_dataloader(train_orig, "Original Dataset (Database/)", num_batches)
    except Exception as e:
        print(f"❌ Error loading original dataset: {e}")
        stats_orig = None
    
    # Create preprocessed dataloaders
    print("\n" + "="*80)
    print("LOADING PREPROCESSED DATASET (no resize)")
    print("="*80)
    
    try:
        train_fast, _, _, _, info_fast = create_fast_dataloaders(
            data_dir='Database_resized/',
            batch_size=batch_size,
            num_workers=num_workers
        )
        
        stats_fast = benchmark_dataloader(train_fast, "Preprocessed Dataset (Database_resized/)", num_batches)
    except Exception as e:
        print(f"❌ Error loading preprocessed dataset: {e}")
        print(f"\nPlease run preprocessing first:")
        print(f"   python scripts/resize_images.py")
        stats_fast = None
    
    # Compare results
    if stats_orig and stats_fast:
        print("\n" + "="*80)
        print("PERFORMANCE COMPARISON RESULTS")
        print("="*80)
        
        print(f"\n📊 Side-by-Side Comparison:")
        print(f"\n{'Metric':<25} {'Original':<20} {'Preprocessed':<20} {'Improvement':<15}")
        print("-" * 80)
        
        # Total time
        speedup = (stats_orig['total_time'] / stats_fast['total_time'] - 1) * 100
        print(f"{'Total Time':<25} {stats_orig['total_time']:>6.2f}s{' '*13} {stats_fast['total_time']:>6.2f}s{' '*13} {speedup:>6.1f}% faster")
        
        # Avg batch time
        speedup = (stats_orig['avg_batch_time'] / stats_fast['avg_batch_time'] - 1) * 100
        print(f"{'Avg Batch Time':<25} {stats_orig['avg_batch_time']*1000:>6.1f}ms{' '*12} {stats_fast['avg_batch_time']*1000:>6.1f}ms{' '*12} {speedup:>6.1f}% faster")
        
        # Throughput
        speedup = (stats_fast['throughput'] / stats_orig['throughput'] - 1) * 100
        print(f"{'Throughput':<25} {stats_orig['throughput']:>6.1f} img/s{' '*9} {stats_fast['throughput']:>6.1f} img/s{' '*9} {speedup:>6.1f}% faster")
        
        # Calculate epoch time savings
        batches_per_epoch = info_orig['train_size'] // batch_size
        orig_epoch_time = stats_orig['avg_batch_time'] * batches_per_epoch / 60
        fast_epoch_time = stats_fast['avg_batch_time'] * batches_per_epoch / 60
        time_saved = orig_epoch_time - fast_epoch_time
        
        print(f"\n⏱️  Estimated Full Epoch Time:")
        print(f"   Original:     {orig_epoch_time:.2f} minutes")
        print(f"   Preprocessed: {fast_epoch_time:.2f} minutes")
        print(f"   Time saved:   {time_saved:.2f} minutes per epoch")
        
        # 50 epochs projection
        print(f"\n🚀 50 Epochs Training Projection:")
        print(f"   Original:     {orig_epoch_time * 50:.1f} minutes ({orig_epoch_time * 50 / 60:.1f} hours)")
        print(f"   Preprocessed: {fast_epoch_time * 50:.1f} minutes ({fast_epoch_time * 50 / 60:.1f} hours)")
        print(f"   ⚡ Total time saved: {time_saved * 50:.1f} minutes ({time_saved * 50 / 60:.1f} hours)")
        
        # Recommendation
        overall_speedup = (stats_orig['total_time'] / stats_fast['total_time'] - 1) * 100
        print(f"\n💡 RECOMMENDATION:")
        if overall_speedup > 5:
            print(f"   ✅ Use preprocessed dataset (Database_resized/)")
            print(f"   ✅ {overall_speedup:.1f}% faster loading")
            print(f"   ✅ {time_saved * 50 / 60:.1f} hours saved over 50 epochs")
        elif overall_speedup > 0:
            print(f"   ⚡ Preprocessed is {overall_speedup:.1f}% faster (marginal improvement)")
            print(f"   → Use either dataset (both work well)")
        else:
            print(f"   ⚠️  No significant improvement detected")
            print(f"   → Check if preprocessing was done correctly")
    
    print("\n" + "="*80)
    print("COMPARISON COMPLETE!")
    print("="*80)


if __name__ == '__main__':
    main()
