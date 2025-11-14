"""
Model Verification Script for CropShield AI
============================================

Verifies model forward pass, output shape, inference speed, and device compatibility
before training phase.

Tests:
1. Random tensor forward pass
2. Output shape validation
3. Inference latency benchmarking
4. CPU vs GPU compatibility
5. Batch size handling
"""

import torch
import torch.nn as nn
import time
from typing import Dict, Any, Optional
import numpy as np

from model_custom_cnn import CropShieldCNN
from model_setup import get_device


def verify_forward_pass(
    model: nn.Module,
    input_shape: tuple = (1, 3, 224, 224),
    expected_output_classes: int = 22,
    device: Optional[torch.device] = None
) -> Dict[str, Any]:
    """
    Verify model forward pass with random input.
    
    Args:
        model: PyTorch model to verify
        input_shape: Input tensor shape (B, C, H, W)
        expected_output_classes: Expected number of output classes
        device: Device to run on (auto-detected if None)
        
    Returns:
        Dictionary with verification results
    """
    if device is None:
        device = next(model.parameters()).device
    
    print("\n" + "="*60)
    print("FORWARD PASS VERIFICATION")
    print("="*60)
    
    # Generate random input
    x = torch.randn(*input_shape).to(device)
    print(f"Input shape:  {list(x.shape)}")
    print(f"Input device: {x.device}")
    print(f"Input dtype:  {x.dtype}")
    
    # Forward pass
    model.eval()
    with torch.no_grad():
        output = model(x)
    
    # Verify output
    batch_size = input_shape[0]
    expected_shape = (batch_size, expected_output_classes)
    
    print(f"\nOutput shape:    {list(output.shape)}")
    print(f"Expected shape:  {list(expected_shape)}")
    print(f"Output device:   {output.device}")
    print(f"Output dtype:    {output.dtype}")
    print(f"Output range:    [{output.min():.3f}, {output.max():.3f}]")
    
    # Check shape
    shape_correct = tuple(output.shape) == expected_shape
    
    if shape_correct:
        print(f"\n✅ Output shape correct: {list(output.shape)}")
    else:
        print(f"\n❌ Output shape incorrect!")
        print(f"   Expected: {list(expected_shape)}")
        print(f"   Got:      {list(output.shape)}")
    
    # Get predicted class
    probs = torch.softmax(output, dim=1)
    pred_class = torch.argmax(probs, dim=1)
    confidence = probs[0, pred_class].item()
    
    print(f"\nPrediction (untrained model):")
    print(f"  Predicted class: {pred_class.item()}")
    print(f"  Confidence:      {confidence*100:.2f}%")
    
    return {
        'input_shape': input_shape,
        'output_shape': tuple(output.shape),
        'expected_shape': expected_shape,
        'shape_correct': shape_correct,
        'output_range': (output.min().item(), output.max().item()),
        'predicted_class': pred_class.item(),
        'confidence': confidence
    }


def benchmark_inference_speed(
    model: nn.Module,
    input_shape: tuple = (1, 3, 224, 224),
    num_iterations: int = 10,
    warmup_iterations: int = 3,
    device: Optional[torch.device] = None
) -> Dict[str, float]:
    """
    Benchmark model inference speed.
    
    Args:
        model: PyTorch model to benchmark
        input_shape: Input tensor shape (B, C, H, W)
        num_iterations: Number of inference iterations
        warmup_iterations: Number of warmup iterations (excluded from timing)
        device: Device to run on (auto-detected if None)
        
    Returns:
        Dictionary with timing statistics (in milliseconds)
    """
    if device is None:
        device = next(model.parameters()).device
    
    print("\n" + "="*60)
    print("INFERENCE SPEED BENCHMARK")
    print("="*60)
    print(f"Device:           {device}")
    print(f"Input shape:      {list(input_shape)}")
    print(f"Warmup iters:     {warmup_iterations}")
    print(f"Benchmark iters:  {num_iterations}")
    
    model.eval()
    
    # Warmup
    print("\n⏳ Warming up...")
    for _ in range(warmup_iterations):
        x = torch.randn(*input_shape).to(device)
        with torch.no_grad():
            _ = model(x)
    
    if device.type == 'cuda':
        torch.cuda.synchronize()
    
    # Benchmark
    print("⏳ Benchmarking...")
    latencies = []
    
    for i in range(num_iterations):
        x = torch.randn(*input_shape).to(device)
        
        if device.type == 'cuda':
            torch.cuda.synchronize()
        
        start_time = time.perf_counter()
        
        with torch.no_grad():
            _ = model(x)
        
        if device.type == 'cuda':
            torch.cuda.synchronize()
        
        end_time = time.perf_counter()
        
        latency_ms = (end_time - start_time) * 1000
        latencies.append(latency_ms)
    
    # Statistics
    latencies = np.array(latencies)
    mean_latency = np.mean(latencies)
    std_latency = np.std(latencies)
    min_latency = np.min(latencies)
    max_latency = np.max(latencies)
    median_latency = np.median(latencies)
    
    print(f"\n📊 Latency Statistics (ms):")
    print(f"  Mean:   {mean_latency:.2f} ms")
    print(f"  Median: {median_latency:.2f} ms")
    print(f"  Std:    {std_latency:.2f} ms")
    print(f"  Min:    {min_latency:.2f} ms")
    print(f"  Max:    {max_latency:.2f} ms")
    print(f"\n📈 Throughput: {1000/mean_latency:.2f} images/sec")
    
    return {
        'mean_ms': mean_latency,
        'median_ms': median_latency,
        'std_ms': std_latency,
        'min_ms': min_latency,
        'max_ms': max_latency,
        'throughput_fps': 1000 / mean_latency
    }


def verify_batch_handling(
    model: nn.Module,
    batch_sizes: list = [1, 4, 8, 16, 32],
    device: Optional[torch.device] = None
) -> Dict[int, Dict[str, Any]]:
    """
    Verify model handles different batch sizes correctly.
    
    Args:
        model: PyTorch model to verify
        batch_sizes: List of batch sizes to test
        device: Device to run on (auto-detected if None)
        
    Returns:
        Dictionary mapping batch size to results
    """
    if device is None:
        device = next(model.parameters()).device
    
    print("\n" + "="*60)
    print("BATCH SIZE VERIFICATION")
    print("="*60)
    
    model.eval()
    results = {}
    
    for batch_size in batch_sizes:
        try:
            x = torch.randn(batch_size, 3, 224, 224).to(device)
            
            with torch.no_grad():
                output = model(x)
            
            success = output.shape[0] == batch_size
            
            print(f"Batch size {batch_size:2d}: ", end="")
            if success:
                print(f"✅ Output shape: {list(output.shape)}")
            else:
                print(f"❌ Shape mismatch: {list(output.shape)}")
            
            results[batch_size] = {
                'success': success,
                'output_shape': tuple(output.shape)
            }
            
        except RuntimeError as e:
            print(f"Batch size {batch_size:2d}: ❌ Error: {str(e)[:50]}...")
            results[batch_size] = {
                'success': False,
                'error': str(e)
            }
    
    return results


def verify_cpu_gpu_compatibility(
    model: nn.Module,
    input_shape: tuple = (1, 3, 224, 224)
) -> Dict[str, Any]:
    """
    Verify model runs on both CPU and GPU (if available).
    
    Args:
        model: PyTorch model to verify
        input_shape: Input tensor shape
        
    Returns:
        Dictionary with compatibility results
    """
    print("\n" + "="*60)
    print("CPU/GPU COMPATIBILITY CHECK")
    print("="*60)
    
    results = {}
    
    # Test CPU
    print("\n📍 Testing on CPU...")
    try:
        model_cpu = model.cpu()
        x_cpu = torch.randn(*input_shape)
        
        model_cpu.eval()
        with torch.no_grad():
            output_cpu = model_cpu(x_cpu)
        
        print(f"✅ CPU inference successful")
        print(f"   Output shape: {list(output_cpu.shape)}")
        
        results['cpu'] = {
            'success': True,
            'output_shape': tuple(output_cpu.shape)
        }
        
    except Exception as e:
        print(f"❌ CPU inference failed: {e}")
        results['cpu'] = {
            'success': False,
            'error': str(e)
        }
    
    # Test GPU
    if torch.cuda.is_available():
        print("\n📍 Testing on GPU...")
        try:
            model_gpu = model.cuda()
            x_gpu = torch.randn(*input_shape).cuda()
            
            model_gpu.eval()
            with torch.no_grad():
                output_gpu = model_gpu(x_gpu)
            
            print(f"✅ GPU inference successful")
            print(f"   Output shape: {list(output_gpu.shape)}")
            print(f"   Device: {output_gpu.device}")
            
            results['gpu'] = {
                'success': True,
                'output_shape': tuple(output_gpu.shape),
                'device': str(output_gpu.device)
            }
            
        except Exception as e:
            print(f"❌ GPU inference failed: {e}")
            results['gpu'] = {
                'success': False,
                'error': str(e)
            }
    else:
        print("\n⚠️  GPU not available")
        results['gpu'] = {
            'success': False,
            'available': False
        }
    
    return results


def compare_cpu_gpu_speed(
    model: nn.Module,
    input_shape: tuple = (1, 3, 224, 224),
    num_iterations: int = 10
) -> Dict[str, Dict[str, float]]:
    """
    Compare inference speed between CPU and GPU.
    
    Args:
        model: PyTorch model to benchmark
        input_shape: Input tensor shape
        num_iterations: Number of iterations for benchmarking
        
    Returns:
        Dictionary with CPU and GPU timing results
    """
    print("\n" + "="*60)
    print("CPU vs GPU SPEED COMPARISON")
    print("="*60)
    
    results = {}
    
    # CPU benchmark
    print("\n📍 Benchmarking CPU...")
    cpu_results = benchmark_inference_speed(
        model.cpu(),
        input_shape=input_shape,
        num_iterations=num_iterations,
        warmup_iterations=3,
        device=torch.device('cpu')
    )
    results['cpu'] = cpu_results
    
    # GPU benchmark
    if torch.cuda.is_available():
        print("\n📍 Benchmarking GPU...")
        gpu_results = benchmark_inference_speed(
            model.cuda(),
            input_shape=input_shape,
            num_iterations=num_iterations,
            warmup_iterations=3,
            device=torch.device('cuda')
        )
        results['gpu'] = gpu_results
        
        # Speedup
        speedup = cpu_results['mean_ms'] / gpu_results['mean_ms']
        print(f"\n⚡ GPU Speedup: {speedup:.2f}x faster than CPU")
        results['speedup'] = speedup
    else:
        print("\n⚠️  GPU not available for comparison")
    
    return results


def run_complete_verification(
    model: nn.Module,
    num_classes: int = 22,
    benchmark_iterations: int = 10
) -> Dict[str, Any]:
    """
    Run complete model verification suite.
    
    Args:
        model: PyTorch model to verify
        num_classes: Expected number of output classes
        benchmark_iterations: Number of iterations for speed benchmarking
        
    Returns:
        Dictionary with all verification results
    """
    print("\n" + "="*60)
    print("🔍 CROPSHIELD AI - COMPLETE MODEL VERIFICATION")
    print("="*60)
    
    all_results = {}
    
    # 1. Forward pass verification
    forward_results = verify_forward_pass(
        model,
        input_shape=(1, 3, 224, 224),
        expected_output_classes=num_classes
    )
    all_results['forward_pass'] = forward_results
    
    # 2. Batch size verification
    batch_results = verify_batch_handling(
        model,
        batch_sizes=[1, 4, 8, 16, 32]
    )
    all_results['batch_handling'] = batch_results
    
    # 3. CPU/GPU compatibility
    compatibility_results = verify_cpu_gpu_compatibility(model)
    all_results['compatibility'] = compatibility_results
    
    # 4. Speed comparison
    if torch.cuda.is_available():
        speed_results = compare_cpu_gpu_speed(
            model,
            num_iterations=benchmark_iterations
        )
        all_results['speed_comparison'] = speed_results
    
    # Summary
    print("\n" + "="*60)
    print("📋 VERIFICATION SUMMARY")
    print("="*60)
    
    print("\n✅ Forward Pass:")
    print(f"   Shape correct:    {forward_results['shape_correct']}")
    print(f"   Output shape:     {list(forward_results['output_shape'])}")
    print(f"   Expected shape:   {list(forward_results['expected_shape'])}")
    
    print("\n✅ Batch Handling:")
    successful_batches = sum(1 for r in batch_results.values() if r['success'])
    print(f"   Successful:       {successful_batches}/{len(batch_results)} batch sizes")
    
    print("\n✅ Device Compatibility:")
    print(f"   CPU:              {'✅' if compatibility_results['cpu']['success'] else '❌'}")
    if 'gpu' in compatibility_results:
        print(f"   GPU:              {'✅' if compatibility_results['gpu']['success'] else '❌'}")
    
    if torch.cuda.is_available() and 'speed_comparison' in all_results:
        print("\n⚡ Performance:")
        print(f"   CPU latency:      {speed_results['cpu']['mean_ms']:.2f} ms")
        print(f"   GPU latency:      {speed_results['gpu']['mean_ms']:.2f} ms")
        print(f"   GPU speedup:      {speed_results['speedup']:.2f}x")
    
    print("\n" + "="*60)
    print("✅ VERIFICATION COMPLETE!")
    print("="*60)
    
    return all_results


# ============================================================
# MAIN EXECUTION
# ============================================================

if __name__ == "__main__":
    print("\n" + "="*60)
    print("🚀 MODEL VERIFICATION - CROPSHIELD AI")
    print("="*60)
    
    # Load model
    print("\n📦 Loading CropShieldCNN...")
    model = CropShieldCNN(num_classes=22)
    
    # Get device
    device = get_device()
    model = model.to(device)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"✅ Model loaded: {total_params:,} parameters")
    
    # Run complete verification
    results = run_complete_verification(
        model=model,
        num_classes=22,
        benchmark_iterations=10
    )
    
    print("\n🎯 Model is ready for training!")
    print("\n💡 Next steps:")
    print("   1. Create training script (train_custom_cnn.py)")
    print("   2. Load data with make_loaders()")
    print("   3. Start training for 100 epochs")
    print("   4. Monitor validation accuracy")
    print("   5. Save best model checkpoint")
