"""
CropShield AI - DataLoader Throughput Benchmark
Comprehensive testing tool to measure and optimize data loading performance

Tests different num_workers configurations and provides optimization recommendations
"""

import torch
import time
import os
from fast_dataset import FastImageFolder, make_loaders
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from collections import defaultdict
import psutil
import platform

# Benchmark configuration
BENCHMARK_BATCHES = 100  # Number of batches to test (or one epoch if smaller)
BATCH_SIZE = 32
TEST_WORKERS = [0, 1, 2, 4, 8, 12]  # Different num_workers to test


def get_system_info():
    """Get system hardware information"""
    info = {
        'os': platform.system(),
        'cpu_count_physical': psutil.cpu_count(logical=False),
        'cpu_count_logical': psutil.cpu_count(logical=True),
        'ram_gb': round(psutil.virtual_memory().total / (1024**3), 1),
        'gpu_available': torch.cuda.is_available(),
        'gpu_name': torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None',
        'cuda_version': torch.version.cuda if torch.cuda.is_available() else 'N/A'
    }
    return info


def benchmark_dataloader(loader, num_batches=100, warmup_batches=5, desc="DataLoader"):
    """
    Benchmark DataLoader throughput
    
    Args:
        loader: DataLoader to benchmark
        num_batches: Number of batches to test
        warmup_batches: Number of warmup batches (not counted)
        desc: Description for display
    
    Returns:
        Dictionary with detailed timing statistics
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Warmup (allows workers to initialize, caches to warm up)
    print(f"\n🔥 Warming up ({warmup_batches} batches)...", end='', flush=True)
    for i, (images, labels) in enumerate(loader):
        if i >= warmup_batches:
            break
        if device.type == 'cuda':
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
    print(" Done!")
    
    # Actual benchmark
    print(f"📊 Benchmarking {desc} ({num_batches} batches)...")
    
    batch_times = []
    total_images = 0
    
    # CPU and memory tracking
    process = psutil.Process()
    cpu_percentages = []
    memory_percentages = []
    
    start_total = time.time()
    
    for i, (images, labels) in enumerate(loader):
        if i >= num_batches:
            break
        
        batch_start = time.time()
        
        # Move to device (simulates training)
        if device.type == 'cuda':
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            torch.cuda.synchronize()  # Wait for GPU transfer to complete
        
        batch_time = time.time() - batch_start
        batch_times.append(batch_time)
        total_images += len(images)
        
        # Track CPU and memory
        try:
            cpu_percentages.append(process.cpu_percent())
            memory_percentages.append(process.memory_percent())
        except:
            pass
        
        # Progress indicator
        if (i + 1) % 10 == 0:
            print(f"   Progress: {i+1}/{num_batches} batches", end='\r', flush=True)
    
    total_time = time.time() - start_total
    print(f"   Progress: {len(batch_times)}/{num_batches} batches - Complete!")
    
    # Calculate statistics
    avg_batch_time = sum(batch_times) / len(batch_times)
    min_batch_time = min(batch_times)
    max_batch_time = max(batch_times)
    std_batch_time = (sum((t - avg_batch_time)**2 for t in batch_times) / len(batch_times))**0.5
    throughput = total_images / total_time
    
    # Calculate percentiles
    sorted_times = sorted(batch_times)
    p50 = sorted_times[len(sorted_times) // 2]
    p95 = sorted_times[int(len(sorted_times) * 0.95)]
    p99 = sorted_times[int(len(sorted_times) * 0.99)]
    
    stats = {
        'total_time': total_time,
        'total_images': total_images,
        'num_batches': len(batch_times),
        'avg_batch_time': avg_batch_time,
        'min_batch_time': min_batch_time,
        'max_batch_time': max_batch_time,
        'std_batch_time': std_batch_time,
        'p50_batch_time': p50,
        'p95_batch_time': p95,
        'p99_batch_time': p99,
        'throughput': throughput,
        'batch_times': batch_times,
        'avg_cpu_percent': sum(cpu_percentages) / len(cpu_percentages) if cpu_percentages else 0,
        'avg_memory_percent': sum(memory_percentages) / len(memory_percentages) if memory_percentages else 0
    }
    
    return stats


def print_benchmark_results(stats, num_workers):
    """Print formatted benchmark results"""
    print(f"\n{'='*80}")
    print(f"RESULTS: num_workers={num_workers}")
    print(f"{'='*80}")
    
    print(f"\n⏱️  Timing:")
    print(f"   Total time:      {stats['total_time']:.2f}s")
    print(f"   Total images:    {stats['total_images']:,}")
    print(f"   Batches tested:  {stats['num_batches']}")
    
    print(f"\n📦 Per-Batch Statistics:")
    print(f"   Average:         {stats['avg_batch_time']*1000:.1f}ms")
    print(f"   Min:             {stats['min_batch_time']*1000:.1f}ms")
    print(f"   Max:             {stats['max_batch_time']*1000:.1f}ms")
    print(f"   Std Dev:         {stats['std_batch_time']*1000:.1f}ms")
    print(f"   P50 (median):    {stats['p50_batch_time']*1000:.1f}ms")
    print(f"   P95:             {stats['p95_batch_time']*1000:.1f}ms")
    print(f"   P99:             {stats['p99_batch_time']*1000:.1f}ms")
    
    print(f"\n🚀 Throughput:")
    print(f"   Images/second:   {stats['throughput']:.1f}")
    print(f"   Batches/second:  {stats['num_batches']/stats['total_time']:.2f}")
    
    if stats['avg_cpu_percent'] > 0:
        print(f"\n💻 Resource Usage:")
        print(f"   Avg CPU:         {stats['avg_cpu_percent']:.1f}%")
        print(f"   Avg Memory:      {stats['avg_memory_percent']:.1f}%")
    
    # Calculate epoch time estimate
    batches_per_epoch = 560  # Approximate for 17,909 training images / 32 batch size
    epoch_time = stats['avg_batch_time'] * batches_per_epoch / 60
    print(f"\n📈 Training Estimates (17,909 training images):")
    print(f"   Epoch time:      {epoch_time:.2f} minutes")
    print(f"   50 epochs:       {epoch_time * 50:.1f} minutes ({epoch_time * 50 / 60:.1f} hours)")


def compare_configurations(results):
    """Compare different num_workers configurations"""
    print("\n" + "="*80)
    print("CONFIGURATION COMPARISON")
    print("="*80)
    
    print(f"\n{'num_workers':<12} {'Throughput':<15} {'Avg Batch':<15} {'CPU %':<10} {'Rating':<10}")
    print("-" * 80)
    
    best_throughput = max(r['throughput'] for r in results.values())
    
    for num_workers in sorted(results.keys()):
        stats = results[num_workers]
        throughput = stats['throughput']
        batch_time = stats['avg_batch_time'] * 1000
        cpu = stats['avg_cpu_percent']
        
        # Rating based on throughput
        if throughput >= best_throughput * 0.95:
            rating = "⭐⭐⭐ Best"
        elif throughput >= best_throughput * 0.85:
            rating = "⭐⭐ Good"
        elif throughput >= best_throughput * 0.70:
            rating = "⭐ OK"
        else:
            rating = "❌ Slow"
        
        print(f"{num_workers:<12} {throughput:>8.1f} img/s   {batch_time:>8.1f}ms      {cpu:>6.1f}%   {rating}")
    
    # Find optimal configuration
    optimal_workers = max(results.keys(), key=lambda k: results[k]['throughput'])
    optimal_throughput = results[optimal_workers]['throughput']
    
    print(f"\n🏆 OPTIMAL CONFIGURATION:")
    print(f"   num_workers: {optimal_workers}")
    print(f"   Throughput: {optimal_throughput:.1f} img/s")
    print(f"   Speedup from worst: {optimal_throughput / min(r['throughput'] for r in results.values()):.2f}x")


def plot_results(results, save_path='benchmark_results.png'):
    """Plot benchmark results"""
    try:
        import matplotlib.pyplot as plt
        
        num_workers_list = sorted(results.keys())
        throughputs = [results[w]['throughput'] for w in num_workers_list]
        avg_times = [results[w]['avg_batch_time'] * 1000 for w in num_workers_list]
        cpu_usage = [results[w]['avg_cpu_percent'] for w in num_workers_list]
        
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 4))
        
        # Throughput
        ax1.plot(num_workers_list, throughputs, marker='o', linewidth=2, markersize=8)
        ax1.set_xlabel('num_workers', fontsize=12)
        ax1.set_ylabel('Throughput (img/s)', fontsize=12)
        ax1.set_title('DataLoader Throughput', fontsize=14, fontweight='bold')
        ax1.grid(True, alpha=0.3)
        ax1.set_xticks(num_workers_list)
        
        # Batch time
        ax2.plot(num_workers_list, avg_times, marker='s', linewidth=2, markersize=8, color='orange')
        ax2.set_xlabel('num_workers', fontsize=12)
        ax2.set_ylabel('Avg Batch Time (ms)', fontsize=12)
        ax2.set_title('Average Batch Loading Time', fontsize=14, fontweight='bold')
        ax2.grid(True, alpha=0.3)
        ax2.set_xticks(num_workers_list)
        
        # CPU usage
        ax3.plot(num_workers_list, cpu_usage, marker='^', linewidth=2, markersize=8, color='green')
        ax3.set_xlabel('num_workers', fontsize=12)
        ax3.set_ylabel('CPU Usage (%)', fontsize=12)
        ax3.set_title('CPU Utilization', fontsize=14, fontweight='bold')
        ax3.grid(True, alpha=0.3)
        ax3.set_xticks(num_workers_list)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"\n📊 Plot saved to: {save_path}")
        
    except ImportError:
        print("\n⚠️  matplotlib not installed, skipping plot")
    except Exception as e:
        print(f"\n⚠️  Error creating plot: {e}")


def interpret_results(results, system_info):
    """Provide interpretation and recommendations"""
    print("\n" + "="*80)
    print("PERFORMANCE INTERPRETATION & RECOMMENDATIONS")
    print("="*80)
    
    # Analyze throughput scaling
    throughputs = [results[w]['throughput'] for w in sorted(results.keys())]
    
    print("\n🔍 Analysis:")
    
    # Check if throughput increases with workers
    if throughputs[-1] > throughputs[0] * 1.5:
        print("   ✅ Good scaling with num_workers")
        print("      → Data loading benefits from parallelization")
        bottleneck = "CPU-bound (decoding/transforms)"
    elif throughputs[-1] > throughputs[0] * 1.1:
        print("   ⚠️  Moderate scaling with num_workers")
        print("      → Some benefit from parallelization")
        bottleneck = "Mixed (CPU + I/O)"
    else:
        print("   ❌ Poor scaling with num_workers")
        print("      → Bottleneck is not in data loading parallelization")
        bottleneck = "I/O-bound (disk/memory)"
    
    # Check CPU usage
    max_cpu = max(r['avg_cpu_percent'] for r in results.values())
    if max_cpu > 80:
        print(f"   ⚠️  High CPU usage ({max_cpu:.0f}%)")
        print("      → CPU is working hard (good utilization)")
    elif max_cpu > 50:
        print(f"   ✅ Moderate CPU usage ({max_cpu:.0f}%)")
        print("      → Balanced workload")
    else:
        print(f"   ℹ️  Low CPU usage ({max_cpu:.0f}%)")
        print("      → CPU not the bottleneck")
    
    # Check variance in batch times
    avg_std = sum(r['std_batch_time'] for r in results.values()) / len(results)
    avg_mean = sum(r['avg_batch_time'] for r in results.values()) / len(results)
    cv = avg_std / avg_mean  # Coefficient of variation
    
    if cv > 0.5:
        print(f"   ⚠️  High variance in batch times (CV={cv:.2f})")
        print("      → Inconsistent loading (disk seeks, cache misses)")
    elif cv > 0.2:
        print(f"   ✅ Moderate variance in batch times (CV={cv:.2f})")
        print("      → Some inconsistency (normal)")
    else:
        print(f"   ✅ Low variance in batch times (CV={cv:.2f})")
        print("      → Very consistent loading")
    
    print(f"\n🎯 Primary Bottleneck: {bottleneck}")
    
    # System-specific recommendations
    print(f"\n💡 RECOMMENDATIONS FOR YOUR SYSTEM:")
    print(f"   System: {system_info['os']}")
    print(f"   CPU: {system_info['cpu_count_logical']} logical cores ({system_info['cpu_count_physical']} physical)")
    print(f"   RAM: {system_info['ram_gb']} GB")
    print(f"   GPU: {system_info['gpu_name']}")
    
    optimal_workers = max(results.keys(), key=lambda k: results[k]['throughput'])
    optimal_throughput = results[optimal_workers]['throughput']
    
    print(f"\n   🏆 Optimal num_workers: {optimal_workers}")
    print(f"   📊 Expected throughput: {optimal_throughput:.1f} img/s")
    
    # Specific recommendations
    if system_info['os'] == 'Windows' and optimal_workers > 0:
        print(f"\n   ⚠️  Note: Windows multiprocessing can be slow")
        print(f"      Consider num_workers=0 if you encounter issues")
    
    if optimal_workers == 0:
        print(f"\n   ℹ️  Single-threaded (num_workers=0) is optimal")
        print(f"      Reasons:")
        print(f"      • Dataset already preprocessed (fast loading)")
        print(f"      • Windows multiprocessing overhead")
        print(f"      • Or: Disk I/O is the bottleneck")
    
    if bottleneck == "I/O-bound (disk/memory)":
        print(f"\n   💾 Storage Recommendations:")
        print(f"      • Move dataset to SSD/NVMe for faster reads")
        print(f"      • Consider caching dataset to RAM")
        print(f"      • Check disk usage during training")
    
    if bottleneck == "CPU-bound (decoding/transforms)":
        print(f"\n   🖥️  CPU Recommendations:")
        print(f"      • Increase num_workers up to {system_info['cpu_count_physical']}")
        print(f"      • Use GPU augmentation libraries (NVIDIA DALI)")
        print(f"      • Reduce augmentation complexity")
    
    # GPU-specific recommendations
    if system_info['gpu_available']:
        print(f"\n   🎮 GPU Recommendations:")
        print(f"      • Use pin_memory=True (already enabled)")
        print(f"      • Use non_blocking=True when moving to GPU")
        print(f"      • Consider mixed precision training (faster)")
        print(f"      • Increase batch_size if memory allows")


def main():
    """Main benchmark pipeline"""
    print("="*80)
    print("CROPSHIELD AI - DATALOADER THROUGHPUT BENCHMARK")
    print("="*80)
    
    # Get system info
    system_info = get_system_info()
    
    print(f"\n🖥️  SYSTEM INFORMATION:")
    print(f"   OS: {system_info['os']}")
    print(f"   CPU Cores: {system_info['cpu_count_logical']} logical ({system_info['cpu_count_physical']} physical)")
    print(f"   RAM: {system_info['ram_gb']} GB")
    print(f"   GPU: {system_info['gpu_name']}")
    print(f"   CUDA: {system_info['cuda_version']}")
    
    # Load dataset
    print(f"\n📁 Loading dataset...")
    try:
        from torchvision import transforms
        transform = transforms.Compose([
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        dataset = FastImageFolder('Database_resized/', transform=transform)
        print(f"   ✅ Loaded {len(dataset):,} images from {len(dataset.classes)} classes")
    except Exception as e:
        print(f"   ❌ Error loading dataset: {e}")
        print(f"\n   Make sure Database_resized/ exists!")
        print(f"   Run: python scripts/resize_images.py")
        return
    
    # Determine test configurations
    available_workers = TEST_WORKERS.copy()
    
    # Filter workers based on CPU cores
    max_useful_workers = system_info['cpu_count_logical']
    available_workers = [w for w in available_workers if w <= max_useful_workers]
    
    print(f"\n⚙️  BENCHMARK CONFIGURATION:")
    print(f"   Batch size: {BATCH_SIZE}")
    print(f"   Test batches: {BENCHMARK_BATCHES}")
    print(f"   Total test images: {BENCHMARK_BATCHES * BATCH_SIZE:,}")
    print(f"   num_workers to test: {available_workers}")
    
    # Confirm
    response = input(f"\n🚀 Start benchmark? This will take ~{len(available_workers) * 2} minutes [Y/n]: ").strip().lower()
    if response and response not in ['y', 'yes']:
        print("❌ Benchmark cancelled.")
        return
    
    # Run benchmarks
    results = {}
    
    for num_workers in available_workers:
        print(f"\n{'='*80}")
        print(f"TESTING: num_workers={num_workers}")
        print(f"{'='*80}")
        
        # Create DataLoader
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        loader = DataLoader(
            dataset,
            batch_size=BATCH_SIZE,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=(device.type == 'cuda'),
            persistent_workers=(num_workers > 0),
            prefetch_factor=2 if num_workers > 0 else None
        )
        
        # Run benchmark
        try:
            stats = benchmark_dataloader(
                loader,
                num_batches=min(BENCHMARK_BATCHES, len(loader)),
                warmup_batches=5,
                desc=f"num_workers={num_workers}"
            )
            results[num_workers] = stats
            print_benchmark_results(stats, num_workers)
        except Exception as e:
            print(f"   ❌ Error with num_workers={num_workers}: {e}")
            continue
    
    # Compare and interpret
    if len(results) > 1:
        compare_configurations(results)
        interpret_results(results, system_info)
        plot_results(results)
    
    # Save results
    print("\n" + "="*80)
    print("BENCHMARK COMPLETE!")
    print("="*80)
    
    # Summary
    if results:
        optimal_workers = max(results.keys(), key=lambda k: results[k]['throughput'])
        optimal_throughput = results[optimal_workers]['throughput']
        
        print(f"\n🏆 FINAL RECOMMENDATION:")
        print(f"   Use num_workers={optimal_workers}")
        print(f"   Expected: {optimal_throughput:.1f} img/s")
        print(f"   Epoch time: ~{results[optimal_workers]['avg_batch_time'] * 560 / 60:.2f} minutes")
        
        print(f"\n💡 Usage in training:")
        print(f"""
from fast_dataset import make_loaders

train_loader, val_loader, test_loader, _, _ = make_loaders(
    data_dir='Database_resized/',
    batch_size={BATCH_SIZE},
    num_workers={optimal_workers}  # ← Optimal value
)
""")


if __name__ == '__main__':
    main()
