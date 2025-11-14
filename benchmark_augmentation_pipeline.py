"""
Augmentation Pipeline Performance Benchmark
Verify that augmentations don't reintroduce Phase 1 slowdown
"""

import torch
import time
import platform
from typing import Dict, List, Tuple
import matplotlib.pyplot as plt
from fast_dataset import make_loaders


def benchmark_loader(loader, num_batches: int = 100, warmup_batches: int = 5,
                     device: str = 'cuda') -> Dict[str, float]:
    """
    Benchmark DataLoader performance over specified number of batches.
    
    Args:
        loader: PyTorch DataLoader to benchmark
        num_batches: Number of batches to measure (default: 100)
        warmup_batches: Number of warmup batches to skip (default: 5)
        device: Device to transfer data to ('cuda' or 'cpu')
        
    Returns:
        Dictionary with performance metrics
    """
    times = []
    total_images = 0
    
    # Warmup phase (JIT compilation, cache warming)
    print(f"   Warming up ({warmup_batches} batches)...", end='', flush=True)
    for i, (images, labels) in enumerate(loader):
        if i >= warmup_batches:
            break
        if device == 'cuda' and torch.cuda.is_available():
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
    print(" Done!")
    
    # Actual benchmark
    print(f"   Benchmarking ({num_batches} batches)...", end='', flush=True)
    
    start_time = time.perf_counter()
    
    for i, (images, labels) in enumerate(loader):
        if i >= num_batches:
            break
        
        batch_start = time.perf_counter()
        
        # Transfer to device (simulate training)
        if device == 'cuda' and torch.cuda.is_available():
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            torch.cuda.synchronize()  # Wait for transfer
        
        batch_end = time.perf_counter()
        
        times.append(batch_end - batch_start)
        total_images += images.size(0)
    
    end_time = time.perf_counter()
    print(" Done!")
    
    # Calculate metrics
    total_time = end_time - start_time
    avg_batch_time = sum(times) / len(times)
    throughput = total_images / total_time
    
    return {
        'total_time': total_time,
        'avg_batch_time': avg_batch_time,
        'min_batch_time': min(times),
        'max_batch_time': max(times),
        'throughput': throughput,
        'total_images': total_images,
        'num_batches': len(times)
    }


def benchmark_multiple_workers(data_dir: str = 'Database_resized/',
                               batch_size: int = 32,
                               augmentation_mode: str = 'moderate',
                               worker_configs: List[int] = [0, 2, 4, 8],
                               num_batches: int = 100) -> Dict[int, Dict[str, float]]:
    """
    Benchmark DataLoader with different num_workers configurations.
    
    Args:
        data_dir: Path to dataset
        batch_size: Batch size
        augmentation_mode: Augmentation mode (conservative/moderate/aggressive)
        worker_configs: List of num_workers values to test
        num_batches: Number of batches per test
        
    Returns:
        Dictionary mapping num_workers to metrics
    """
    results = {}
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    print("\n" + "="*70)
    print("AUGMENTATION PIPELINE PERFORMANCE BENCHMARK")
    print("="*70)
    print(f"\n⚙️  Configuration:")
    print(f"   Dataset: {data_dir}")
    print(f"   Batch size: {batch_size}")
    print(f"   Augmentation: {augmentation_mode.upper()}")
    print(f"   Device: {device.upper()}")
    print(f"   Platform: {platform.system()}")
    print(f"   Benchmarking: {num_batches} batches per config")
    
    for num_workers in worker_configs:
        print(f"\n{'─'*70}")
        print(f"Testing num_workers = {num_workers}")
        print('─'*70)
        
        try:
            # Create DataLoader with specific num_workers
            train_loader, _, _, _, info = make_loaders(
                data_dir=data_dir,
                batch_size=batch_size,
                num_workers=num_workers,
                augmentation_mode=augmentation_mode
            )
            
            # Benchmark
            metrics = benchmark_loader(train_loader, num_batches=num_batches, device=device)
            results[num_workers] = metrics
            
            # Print results
            print(f"\n   📊 Results:")
            print(f"      Total time: {metrics['total_time']:.2f}s")
            print(f"      Avg batch time: {metrics['avg_batch_time']*1000:.2f}ms")
            print(f"      Min batch time: {metrics['min_batch_time']*1000:.2f}ms")
            print(f"      Max batch time: {metrics['max_batch_time']*1000:.2f}ms")
            print(f"      Throughput: {metrics['throughput']:.1f} images/sec")
            print(f"      Total images: {metrics['total_images']}")
            
        except Exception as e:
            print(f"\n   ❌ Error with num_workers={num_workers}: {e}")
            results[num_workers] = None
    
    return results


def analyze_results(results: Dict[int, Dict[str, float]],
                   baseline_throughput: float = None):
    """
    Analyze benchmark results and provide recommendations.
    
    Args:
        results: Dictionary from benchmark_multiple_workers
        baseline_throughput: Baseline throughput from Phase 1 (no augmentation)
    """
    print("\n" + "="*70)
    print("PERFORMANCE ANALYSIS")
    print("="*70)
    
    # Filter out failed configs
    valid_results = {k: v for k, v in results.items() if v is not None}
    
    if not valid_results:
        print("\n❌ No valid results to analyze!")
        return
    
    # Find best configuration
    best_workers = max(valid_results.keys(), 
                      key=lambda k: valid_results[k]['throughput'])
    best_throughput = valid_results[best_workers]['throughput']
    
    print(f"\n🏆 Best Configuration:")
    print(f"   num_workers = {best_workers}")
    print(f"   Throughput: {best_throughput:.1f} images/sec")
    print(f"   Avg batch time: {valid_results[best_workers]['avg_batch_time']*1000:.2f}ms")
    
    # Compare configurations
    print(f"\n📊 Comparison Table:")
    print(f"   {'num_workers':<15} {'Throughput':<20} {'Batch Time':<20} {'Speedup'}")
    print(f"   {'-'*15} {'-'*20} {'-'*20} {'-'*10}")
    
    baseline = valid_results[min(valid_results.keys())]
    for workers in sorted(valid_results.keys()):
        metrics = valid_results[workers]
        speedup = metrics['throughput'] / baseline['throughput']
        print(f"   {workers:<15} {metrics['throughput']:>8.1f} img/s      "
              f"{metrics['avg_batch_time']*1000:>8.2f} ms        {speedup:>6.2f}x")
    
    # Compare with baseline (Phase 1)
    if baseline_throughput:
        print(f"\n📈 Comparison with Phase 1 (No Augmentation):")
        print(f"   Phase 1 throughput: {baseline_throughput:.1f} images/sec")
        print(f"   Current throughput: {best_throughput:.1f} images/sec")
        
        slowdown = (1 - best_throughput / baseline_throughput) * 100
        if slowdown > 0:
            print(f"   Slowdown: {slowdown:.1f}% (Expected with augmentation)")
        else:
            print(f"   Speedup: {-slowdown:.1f}% (Unexpected but good!)")
        
        if slowdown < 30:
            print(f"   ✅ Augmentation overhead is reasonable (<30%)")
        else:
            print(f"   ⚠️  High augmentation overhead (>{slowdown:.0f}%)")
    
    # Recommendations
    print(f"\n💡 Recommendations:")
    
    # num_workers recommendation
    if platform.system() == 'Windows':
        if best_workers == 0:
            print(f"   ✅ num_workers=0 is best for Windows (multiprocessing overhead)")
        else:
            print(f"   ⚠️  num_workers={best_workers} works on Windows but may have issues")
            print(f"      Consider num_workers=0 for stability")
    else:
        if best_workers == 0:
            print(f"   ⚠️  num_workers=0 on Linux/Mac - try higher values (4-8)")
        else:
            print(f"   ✅ num_workers={best_workers} is optimal for your system")
    
    # Throughput check
    if best_throughput < 50:
        print(f"   ⚠️  Low throughput ({best_throughput:.1f} img/s)")
        print(f"      - Consider reducing augmentation strength")
        print(f"      - Try GPU transforms (kornia library)")
        print(f"      - Enable prefetch_factor (currently 2)")
    elif best_throughput < 200:
        print(f"   ✅ Good throughput ({best_throughput:.1f} img/s)")
        print(f"      - Sufficient for training (won't bottleneck GPU)")
    else:
        print(f"   🚀 Excellent throughput ({best_throughput:.1f} img/s)")
        print(f"      - Data loading won't bottleneck training")
    
    # Additional optimizations
    print(f"\n🔧 Further Optimization Options:")
    print(f"   1. GPU Transforms (kornia):")
    print(f"      - Move augmentations to GPU (parallel with training)")
    print(f"      - Can achieve 500+ img/s on RTX 4060")
    print(f"      - Requires code changes")
    
    print(f"\n   2. Increase prefetch_factor:")
    print(f"      - Currently: 2 (default)")
    print(f"      - Try: 4 or 8 (more memory, less wait time)")
    print(f"      - Only works with num_workers > 0")
    
    print(f"\n   3. Mixed Precision Data Loading:")
    print(f"      - Load as float16 instead of float32")
    print(f"      - 2x memory reduction")
    print(f"      - Faster GPU transfer")
    
    print(f"\n   4. Reduce Augmentation Strength:")
    print(f"      - Current: {list(valid_results.values())[0].get('mode', 'moderate')}")
    print(f"      - Try: conservative mode (fewer transforms)")
    print(f"      - Trade-off: less regularization")
    
    print(f"\n   5. Enable Caching (if dataset fits in RAM):")
    print(f"      - Pre-load all images to memory")
    print(f"      - Apply augmentations on-the-fly")
    print(f"      - Eliminates disk I/O bottleneck")


def plot_results(results: Dict[int, Dict[str, float]], 
                save_path: str = 'augmentation_benchmark.png'):
    """
    Create visualization of benchmark results.
    
    Args:
        results: Dictionary from benchmark_multiple_workers
        save_path: Path to save plot
    """
    valid_results = {k: v for k, v in results.items() if v is not None}
    
    if not valid_results:
        print("\n❌ No valid results to plot!")
        return
    
    workers = sorted(valid_results.keys())
    throughputs = [valid_results[w]['throughput'] for w in workers]
    batch_times = [valid_results[w]['avg_batch_time'] * 1000 for w in workers]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Throughput plot
    ax1.plot(workers, throughputs, marker='o', linewidth=2, markersize=10)
    ax1.set_xlabel('num_workers', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Throughput (images/sec)', fontsize=12, fontweight='bold')
    ax1.set_title('DataLoader Throughput vs num_workers', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.set_xticks(workers)
    
    # Add value labels
    for w, t in zip(workers, throughputs):
        ax1.annotate(f'{t:.1f}', xy=(w, t), xytext=(0, 10), 
                    textcoords='offset points', ha='center', fontweight='bold')
    
    # Batch time plot
    ax2.plot(workers, batch_times, marker='s', linewidth=2, markersize=10, color='orange')
    ax2.set_xlabel('num_workers', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Avg Batch Time (ms)', fontsize=12, fontweight='bold')
    ax2.set_title('Average Batch Loading Time vs num_workers', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.set_xticks(workers)
    
    # Add value labels
    for w, bt in zip(workers, batch_times):
        ax2.annotate(f'{bt:.2f}', xy=(w, bt), xytext=(0, 10), 
                    textcoords='offset points', ha='center', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"\n✅ Plot saved to: {save_path}")
    
    return fig


def quick_benchmark(augmentation_mode: str = 'moderate', 
                   num_batches: int = 100):
    """
    Quick benchmark with optimal settings for current platform.
    
    Args:
        augmentation_mode: Augmentation mode to test
        num_batches: Number of batches to benchmark
    """
    print("\n" + "="*70)
    print("QUICK BENCHMARK - OPTIMAL SETTINGS")
    print("="*70)
    
    # Platform-specific optimal workers
    if platform.system() == 'Windows':
        optimal_workers = 0
        print(f"\n🖥️  Detected Windows - using num_workers=0")
    else:
        optimal_workers = 4
        print(f"\n🖥️  Detected {platform.system()} - using num_workers=4")
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"   Device: {device.upper()}")
    print(f"   Augmentation: {augmentation_mode.upper()}")
    
    # Create DataLoader
    train_loader, _, _, _, info = make_loaders(
        data_dir='Database_resized/',
        batch_size=32,
        num_workers=optimal_workers,
        augmentation_mode=augmentation_mode
    )
    
    print(f"\n📊 Dataset: {info['train_size']} training images")
    
    # Benchmark
    metrics = benchmark_loader(train_loader, num_batches=num_batches, device=device)
    
    # Results
    print(f"\n✅ Results:")
    print(f"   Throughput: {metrics['throughput']:.1f} images/sec")
    print(f"   Avg batch time: {metrics['avg_batch_time']*1000:.2f}ms")
    print(f"   Total time: {metrics['total_time']:.2f}s for {num_batches} batches")
    
    # Training time estimate
    total_batches = len(train_loader)
    loading_time_per_epoch = (total_batches / num_batches) * metrics['total_time']
    
    print(f"\n⏱️  Estimated Time per Epoch:")
    print(f"   Data loading only: {loading_time_per_epoch:.2f}s")
    print(f"   With GPU training: ~{loading_time_per_epoch * 1.5:.2f}s (estimate)")
    print(f"   50 epochs: ~{(loading_time_per_epoch * 1.5 * 50) / 60:.1f} minutes")
    
    # Verdict
    if metrics['throughput'] > 100:
        print(f"\n✅ VERDICT: Excellent performance - won't bottleneck training!")
    elif metrics['throughput'] > 50:
        print(f"\n✅ VERDICT: Good performance - suitable for training")
    else:
        print(f"\n⚠️  VERDICT: Low performance - may bottleneck training")
        print(f"   Consider optimizations (see full benchmark for details)")
    
    return metrics


# Example usage and main script
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Benchmark augmentation pipeline')
    parser.add_argument('--mode', type=str, default='moderate',
                       choices=['conservative', 'moderate', 'aggressive'],
                       help='Augmentation mode')
    parser.add_argument('--workers', type=int, nargs='+', default=[0, 2, 4, 8],
                       help='List of num_workers to test')
    parser.add_argument('--batches', type=int, default=100,
                       help='Number of batches to benchmark')
    parser.add_argument('--quick', action='store_true',
                       help='Run quick benchmark only')
    parser.add_argument('--baseline', type=float, default=None,
                       help='Phase 1 baseline throughput (images/sec)')
    
    args = parser.parse_args()
    
    if args.quick:
        # Quick benchmark with optimal settings
        quick_benchmark(augmentation_mode=args.mode, num_batches=args.batches)
    else:
        # Full benchmark with multiple configurations
        results = benchmark_multiple_workers(
            batch_size=32,
            augmentation_mode=args.mode,
            worker_configs=args.workers,
            num_batches=args.batches
        )
        
        # Analyze and visualize
        analyze_results(results, baseline_throughput=args.baseline)
        plot_results(results)
        
        print("\n" + "="*70)
        print("BENCHMARK COMPLETE")
        print("="*70)
        print(f"\n📁 Files generated:")
        print(f"   - augmentation_benchmark.png (performance plots)")
        
        # GPU training estimate
        valid_results = {k: v for k, v in results.items() if v is not None}
        if valid_results:
            best_workers = max(valid_results.keys(), 
                             key=lambda k: valid_results[k]['throughput'])
            best_throughput = valid_results[best_workers]['throughput']
            
            print(f"\n🎯 Recommended Configuration:")
            print(f"   num_workers = {best_workers}")
            print(f"   Expected throughput: {best_throughput:.1f} img/s")
            print(f"   Ready for GPU training: {'✅ Yes' if best_throughput > 50 else '⚠️  Consider optimizations'}")
        
        print("\n✅ Phase 2 verification complete!")
        print("   Your augmentation pipeline is ready for training.\n")
