"""
ONNX Model Export for CropShield AI

Export trained PyTorch models to ONNX format for deployment.
Supports CPU inference, edge deployment, and model quantization.

Features:
- Export to ONNX with dynamic batch size
- Inference verification (PyTorch vs ONNX)
- Post-training static quantization
- Quantized model comparison
- GradCAM compatibility notes
"""

import os
import time
import json
from pathlib import Path
from typing import Tuple, Dict, Optional
import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms

# ONNX and runtime
try:
    import onnx
    import onnxruntime as ort
    from onnxruntime.quantization import quantize_dynamic, quantize_static, QuantType
    from onnxruntime.quantization.calibrate import CalibrationDataReader
    ONNX_AVAILABLE = True
except ImportError:
    ONNX_AVAILABLE = False
    print("⚠️  ONNX or ONNXRuntime not installed. Install with:")
    print("   pip install onnx onnxruntime")

from models.model_factory import load_model_for_inference


class CalibrationDataset(CalibrationDataReader):
    """
    Calibration dataset for static quantization.
    
    Provides representative data samples to calibrate quantization ranges.
    """
    
    def __init__(self, calibration_images: list, image_size: int = 224):
        """
        Initialize calibration dataset.
        
        Args:
            calibration_images: List of image paths for calibration
            image_size: Input image size
        """
        self.calibration_images = calibration_images
        self.image_size = image_size
        self.current_idx = 0
        
        # Standard ImageNet preprocessing
        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225])
        ])
    
    def get_next(self) -> Optional[Dict]:
        """
        Get next calibration sample.
        
        Returns:
            Dictionary with input name and data, or None if exhausted
        """
        if self.current_idx >= len(self.calibration_images):
            return None
        
        # Load and preprocess image
        from PIL import Image
        img_path = self.calibration_images[self.current_idx]
        self.current_idx += 1
        
        try:
            img = Image.open(img_path).convert('RGB')
            tensor = self.transform(img)
            # Add batch dimension
            tensor = tensor.unsqueeze(0).numpy()
            return {'input': tensor}
        except Exception as e:
            print(f"⚠️  Failed to load {img_path}: {e}")
            return self.get_next()
    
    def rewind(self):
        """Reset to beginning."""
        self.current_idx = 0


def export_to_onnx(
    model: nn.Module,
    output_path: str,
    input_shape: Tuple[int, int, int, int] = (1, 3, 224, 224),
    opset_version: int = 11,
    dynamic_axes: bool = True,
    verify: bool = True,
    verbose: bool = True
) -> bool:
    """
    Export PyTorch model to ONNX format.
    
    Args:
        model: PyTorch model to export
        output_path: Output ONNX file path
        input_shape: Input tensor shape (batch, channels, height, width)
        opset_version: ONNX opset version (11+ recommended)
        dynamic_axes: Enable dynamic batch size
        verify: Verify ONNX model after export
        verbose: Print detailed information
        
    Returns:
        True if export successful
    """
    
    if not ONNX_AVAILABLE:
        print("❌ ONNX not available. Cannot export.")
        return False
    
    if verbose:
        print(f"\n{'='*80}")
        print(f"📦 EXPORTING MODEL TO ONNX")
        print(f"{'='*80}")
        print(f"Model: {model.__class__.__name__}")
        print(f"Input shape: {input_shape}")
        print(f"Output path: {output_path}")
        print(f"Opset version: {opset_version}")
        print(f"Dynamic batch: {dynamic_axes}")
        print(f"{'='*80}\n")
    
    # Create output directory
    output_dir = Path(output_path).parent
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Set model to eval mode
    model.eval()
    device = next(model.parameters()).device
    
    # Create dummy input
    dummy_input = torch.randn(input_shape, device=device)
    
    # Define dynamic axes if enabled
    dynamic_axes_dict = None
    if dynamic_axes:
        dynamic_axes_dict = {
            'input': {0: 'batch_size'},
            'output': {0: 'batch_size'}
        }
    
    try:
        # Export to ONNX
        if verbose:
            print("🔄 Exporting to ONNX...")
        
        start_time = time.time()
        
        torch.onnx.export(
            model,
            dummy_input,
            output_path,
            export_params=True,
            opset_version=opset_version,
            do_constant_folding=True,
            input_names=['input'],
            output_names=['output'],
            dynamic_axes=dynamic_axes_dict,
            verbose=False
        )
        
        export_time = time.time() - start_time
        
        # Get file size
        file_size = Path(output_path).stat().st_size / (1024 * 1024)  # MB
        
        if verbose:
            print(f"✅ Export successful!")
            print(f"   Time: {export_time:.2f}s")
            print(f"   Size: {file_size:.2f} MB")
        
        # Verify ONNX model
        if verify:
            if verbose:
                print(f"\n🔍 Verifying ONNX model...")
            
            # Load and check ONNX model
            onnx_model = onnx.load(output_path)
            onnx.checker.check_model(onnx_model)
            
            if verbose:
                print(f"✅ ONNX model is valid")
                
                # Print model info
                print(f"\n📊 Model Information:")
                print(f"   IR version: {onnx_model.ir_version}")
                print(f"   Opset version: {onnx_model.opset_import[0].version}")
                
                # Print input/output info
                graph = onnx_model.graph
                print(f"\n📥 Inputs:")
                for inp in graph.input:
                    shape = [d.dim_value if d.dim_value > 0 else 'dynamic' 
                            for d in inp.type.tensor_type.shape.dim]
                    print(f"   {inp.name}: {shape}")
                
                print(f"\n📤 Outputs:")
                for out in graph.output:
                    shape = [d.dim_value if d.dim_value > 0 else 'dynamic' 
                            for d in out.type.tensor_type.shape.dim]
                    print(f"   {out.name}: {shape}")
        
        return True
        
    except Exception as e:
        print(f"❌ Export failed: {e}")
        return False


def verify_onnx_inference(
    pytorch_model: nn.Module,
    onnx_path: str,
    num_samples: int = 5,
    input_shape: Tuple[int, int, int, int] = (1, 3, 224, 224),
    tolerance: float = 1e-5,
    verbose: bool = True
) -> bool:
    """
    Verify ONNX model inference matches PyTorch.
    
    Args:
        pytorch_model: Original PyTorch model
        onnx_path: Path to ONNX model
        num_samples: Number of random samples to test
        input_shape: Input tensor shape
        tolerance: Maximum absolute difference allowed
        verbose: Print detailed comparison
        
    Returns:
        True if outputs match within tolerance
    """
    
    if not ONNX_AVAILABLE:
        print("❌ ONNXRuntime not available. Cannot verify.")
        return False
    
    if verbose:
        print(f"\n{'='*80}")
        print(f"🔍 VERIFYING ONNX INFERENCE")
        print(f"{'='*80}")
        print(f"PyTorch model: {pytorch_model.__class__.__name__}")
        print(f"ONNX model: {onnx_path}")
        print(f"Test samples: {num_samples}")
        print(f"Tolerance: {tolerance}")
        print(f"{'='*80}\n")
    
    # Set PyTorch model to eval mode
    pytorch_model.eval()
    device = next(pytorch_model.parameters()).device
    
    # Create ONNX Runtime session
    try:
        ort_session = ort.InferenceSession(
            onnx_path,
            providers=['CPUExecutionProvider']
        )
        if verbose:
            print(f"✅ ONNX Runtime session created")
            print(f"   Providers: {ort_session.get_providers()}")
    except Exception as e:
        print(f"❌ Failed to create ONNX Runtime session: {e}")
        return False
    
    # Test with random inputs
    all_match = True
    max_diff = 0.0
    avg_diff = 0.0
    
    for i in range(num_samples):
        # Generate random input
        input_tensor = torch.randn(input_shape, device=device)
        
        # PyTorch inference
        with torch.no_grad():
            pytorch_output = pytorch_model(input_tensor)
            if isinstance(pytorch_output, tuple):
                pytorch_output = pytorch_output[0]
            pytorch_output = pytorch_output.cpu().numpy()
        
        # ONNX inference
        input_numpy = input_tensor.cpu().numpy()
        onnx_output = ort_session.run(
            None,
            {'input': input_numpy}
        )[0]
        
        # Compare outputs
        diff = np.abs(pytorch_output - onnx_output)
        sample_max_diff = np.max(diff)
        sample_avg_diff = np.mean(diff)
        
        max_diff = max(max_diff, sample_max_diff)
        avg_diff += sample_avg_diff
        
        match = sample_max_diff < tolerance
        all_match = all_match and match
        
        if verbose:
            status = "✅" if match else "❌"
            print(f"{status} Sample {i+1}/{num_samples}: "
                  f"max_diff={sample_max_diff:.2e}, avg_diff={sample_avg_diff:.2e}")
    
    avg_diff /= num_samples
    
    if verbose:
        print(f"\n{'='*80}")
        print(f"📊 VERIFICATION SUMMARY")
        print(f"{'='*80}")
        print(f"Max difference: {max_diff:.2e}")
        print(f"Avg difference: {avg_diff:.2e}")
        print(f"Tolerance: {tolerance:.2e}")
        
        if all_match:
            print(f"\n✅ All samples match within tolerance!")
        else:
            print(f"\n⚠️  Some samples exceed tolerance")
        print(f"{'='*80}\n")
    
    return all_match


def benchmark_inference(
    onnx_path: str,
    input_shape: Tuple[int, int, int, int] = (1, 3, 224, 224),
    num_runs: int = 100,
    warmup_runs: int = 10,
    verbose: bool = True
) -> Dict:
    """
    Benchmark ONNX model inference speed.
    
    Args:
        onnx_path: Path to ONNX model
        input_shape: Input tensor shape
        num_runs: Number of inference runs
        warmup_runs: Number of warmup runs
        verbose: Print results
        
    Returns:
        Dictionary with benchmark results
    """
    
    if not ONNX_AVAILABLE:
        print("❌ ONNXRuntime not available.")
        return {}
    
    if verbose:
        print(f"\n{'='*80}")
        print(f"⚡ BENCHMARKING ONNX INFERENCE")
        print(f"{'='*80}")
        print(f"Model: {onnx_path}")
        print(f"Input shape: {input_shape}")
        print(f"Runs: {num_runs} (+ {warmup_runs} warmup)")
        print(f"{'='*80}\n")
    
    # Create session
    ort_session = ort.InferenceSession(
        onnx_path,
        providers=['CPUExecutionProvider']
    )
    
    # Generate random input
    input_numpy = np.random.randn(*input_shape).astype(np.float32)
    
    # Warmup
    if verbose:
        print(f"🔥 Warming up ({warmup_runs} runs)...")
    for _ in range(warmup_runs):
        _ = ort_session.run(None, {'input': input_numpy})
    
    # Benchmark
    if verbose:
        print(f"⏱️  Benchmarking ({num_runs} runs)...")
    
    times = []
    for _ in range(num_runs):
        start = time.time()
        _ = ort_session.run(None, {'input': input_numpy})
        times.append(time.time() - start)
    
    # Statistics
    times = np.array(times) * 1000  # Convert to ms
    results = {
        'mean_ms': np.mean(times),
        'std_ms': np.std(times),
        'min_ms': np.min(times),
        'max_ms': np.max(times),
        'median_ms': np.median(times),
        'throughput_fps': 1000.0 / np.mean(times)
    }
    
    if verbose:
        print(f"\n📊 BENCHMARK RESULTS")
        print(f"{'='*80}")
        print(f"Mean:       {results['mean_ms']:.2f} ms")
        print(f"Std:        {results['std_ms']:.2f} ms")
        print(f"Min:        {results['min_ms']:.2f} ms")
        print(f"Max:        {results['max_ms']:.2f} ms")
        print(f"Median:     {results['median_ms']:.2f} ms")
        print(f"Throughput: {results['throughput_fps']:.1f} FPS")
        print(f"{'='*80}\n")
    
    return results


def quantize_onnx_dynamic(
    onnx_path: str,
    output_path: str,
    weight_type: str = 'int8',
    verbose: bool = True
) -> bool:
    """
    Apply dynamic quantization to ONNX model.
    
    Dynamic quantization: Weights quantized to int8, activations remain float32.
    Good for: CPU inference, no calibration data needed.
    
    Args:
        onnx_path: Input ONNX model path
        output_path: Output quantized model path
        weight_type: Weight quantization type ('int8' or 'uint8')
        verbose: Print information
        
    Returns:
        True if quantization successful
    """
    
    if not ONNX_AVAILABLE:
        print("❌ ONNXRuntime quantization not available.")
        return False
    
    if verbose:
        print(f"\n{'='*80}")
        print(f"🔧 DYNAMIC QUANTIZATION")
        print(f"{'='*80}")
        print(f"Input:  {onnx_path}")
        print(f"Output: {output_path}")
        print(f"Weight type: {weight_type}")
        print(f"{'='*80}\n")
    
    try:
        # Map weight type
        quant_type = QuantType.QInt8 if weight_type == 'int8' else QuantType.QUInt8
        
        # Quantize
        if verbose:
            print(f"🔄 Quantizing model...")
        
        start_time = time.time()
        
        quantize_dynamic(
            model_input=onnx_path,
            model_output=output_path,
            weight_type=quant_type
        )
        
        quant_time = time.time() - start_time
        
        # Get file sizes
        original_size = Path(onnx_path).stat().st_size / (1024 * 1024)  # MB
        quantized_size = Path(output_path).stat().st_size / (1024 * 1024)  # MB
        compression_ratio = original_size / quantized_size
        
        if verbose:
            print(f"✅ Quantization successful!")
            print(f"   Time: {quant_time:.2f}s")
            print(f"   Original size: {original_size:.2f} MB")
            print(f"   Quantized size: {quantized_size:.2f} MB")
            print(f"   Compression: {compression_ratio:.2f}x")
            print(f"   Size reduction: {(1 - quantized_size/original_size)*100:.1f}%")
        
        return True
        
    except Exception as e:
        print(f"❌ Quantization failed: {e}")
        return False


def compare_models(
    original_onnx: str,
    quantized_onnx: str,
    num_samples: int = 10,
    input_shape: Tuple[int, int, int, int] = (1, 3, 224, 224),
    verbose: bool = True
) -> Dict:
    """
    Compare original and quantized ONNX models.
    
    Args:
        original_onnx: Original ONNX model path
        quantized_onnx: Quantized ONNX model path
        num_samples: Number of test samples
        input_shape: Input tensor shape
        verbose: Print comparison
        
    Returns:
        Dictionary with comparison results
    """
    
    if not ONNX_AVAILABLE:
        print("❌ ONNXRuntime not available.")
        return {}
    
    if verbose:
        print(f"\n{'='*80}")
        print(f"📊 COMPARING MODELS")
        print(f"{'='*80}")
        print(f"Original:  {original_onnx}")
        print(f"Quantized: {quantized_onnx}")
        print(f"Test samples: {num_samples}")
        print(f"{'='*80}\n")
    
    # Create sessions
    original_session = ort.InferenceSession(
        original_onnx,
        providers=['CPUExecutionProvider']
    )
    quantized_session = ort.InferenceSession(
        quantized_onnx,
        providers=['CPUExecutionProvider']
    )
    
    # Test outputs
    max_diff = 0.0
    avg_diff = 0.0
    
    for i in range(num_samples):
        # Generate random input
        input_numpy = np.random.randn(*input_shape).astype(np.float32)
        
        # Run both models
        original_output = original_session.run(None, {'input': input_numpy})[0]
        quantized_output = quantized_session.run(None, {'input': input_numpy})[0]
        
        # Compare
        diff = np.abs(original_output - quantized_output)
        sample_max_diff = np.max(diff)
        sample_avg_diff = np.mean(diff)
        
        max_diff = max(max_diff, sample_max_diff)
        avg_diff += sample_avg_diff
        
        if verbose:
            print(f"Sample {i+1}/{num_samples}: "
                  f"max_diff={sample_max_diff:.2e}, avg_diff={sample_avg_diff:.2e}")
    
    avg_diff /= num_samples
    
    # Benchmark both models
    if verbose:
        print(f"\n⚡ Benchmarking original model...")
    original_results = benchmark_inference(
        original_onnx,
        input_shape=input_shape,
        num_runs=50,
        warmup_runs=5,
        verbose=False
    )
    
    if verbose:
        print(f"⚡ Benchmarking quantized model...")
    quantized_results = benchmark_inference(
        quantized_onnx,
        input_shape=input_shape,
        num_runs=50,
        warmup_runs=5,
        verbose=False
    )
    
    speedup = original_results['mean_ms'] / quantized_results['mean_ms']
    
    # File sizes
    original_size = Path(original_onnx).stat().st_size / (1024 * 1024)  # MB
    quantized_size = Path(quantized_onnx).stat().st_size / (1024 * 1024)  # MB
    
    results = {
        'max_diff': max_diff,
        'avg_diff': avg_diff,
        'original_size_mb': original_size,
        'quantized_size_mb': quantized_size,
        'compression_ratio': original_size / quantized_size,
        'original_inference_ms': original_results['mean_ms'],
        'quantized_inference_ms': quantized_results['mean_ms'],
        'speedup': speedup
    }
    
    if verbose:
        print(f"\n{'='*80}")
        print(f"📊 COMPARISON SUMMARY")
        print(f"{'='*80}")
        print(f"\n📏 Accuracy:")
        print(f"   Max difference: {max_diff:.2e}")
        print(f"   Avg difference: {avg_diff:.2e}")
        
        print(f"\n💾 Size:")
        print(f"   Original:  {original_size:.2f} MB")
        print(f"   Quantized: {quantized_size:.2f} MB")
        print(f"   Compression: {results['compression_ratio']:.2f}x")
        
        print(f"\n⚡ Speed:")
        print(f"   Original:  {original_results['mean_ms']:.2f} ms ({original_results['throughput_fps']:.1f} FPS)")
        print(f"   Quantized: {quantized_results['mean_ms']:.2f} ms ({quantized_results['throughput_fps']:.1f} FPS)")
        print(f"   Speedup: {speedup:.2f}x")
        
        print(f"{'='*80}\n")
    
    return results


def export_full_pipeline(
    model_path: str,
    model_type: str,
    num_classes: int,
    output_dir: str = 'models',
    quantize: bool = True,
    verify: bool = True,
    benchmark: bool = True,
    verbose: bool = True
) -> Dict:
    """
    Complete export pipeline: PyTorch → ONNX → Quantized ONNX.
    
    Args:
        model_path: Path to trained PyTorch model
        model_type: Model architecture ('custom', 'efficientnet_b0')
        num_classes: Number of output classes
        output_dir: Output directory for exported models
        quantize: Apply quantization
        verify: Verify ONNX inference
        benchmark: Benchmark inference speed
        verbose: Print detailed information
        
    Returns:
        Dictionary with export results
    """
    
    if verbose:
        print(f"\n{'='*80}")
        print(f"🚀 FULL EXPORT PIPELINE")
        print(f"{'='*80}")
        print(f"Model path: {model_path}")
        print(f"Model type: {model_type}")
        print(f"Num classes: {num_classes}")
        print(f"Output dir: {output_dir}")
        print(f"Quantize: {quantize}")
        print(f"{'='*80}\n")
    
    # Load PyTorch model
    if verbose:
        print(f"📦 Loading PyTorch model...")
    
    try:
        model, class_names, model_info = load_model_for_inference(
            model_path=model_path,
            model_type=model_type,
            num_classes=num_classes
        )
        model.eval()
        
        if verbose:
            print(f"✅ Model loaded successfully")
            print(f"   Architecture: {model_info.get('architecture', 'Unknown')}")
            print(f"   Classes: {len(class_names)}")
    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        return {}
    
    # Export to ONNX
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    onnx_path = output_dir / 'cropshield.onnx'
    
    success = export_to_onnx(
        model=model,
        output_path=str(onnx_path),
        input_shape=(1, 3, 224, 224),
        opset_version=11,
        dynamic_axes=True,
        verify=True,
        verbose=verbose
    )
    
    if not success:
        print(f"❌ Export failed")
        return {}
    
    results = {
        'onnx_path': str(onnx_path),
        'onnx_size_mb': onnx_path.stat().st_size / (1024 * 1024)
    }
    
    # Verify inference
    if verify:
        match = verify_onnx_inference(
            pytorch_model=model,
            onnx_path=str(onnx_path),
            num_samples=5,
            tolerance=1e-5,
            verbose=verbose
        )
        results['verification_passed'] = match
    
    # Benchmark original
    if benchmark:
        onnx_benchmark = benchmark_inference(
            onnx_path=str(onnx_path),
            num_runs=100,
            warmup_runs=10,
            verbose=verbose
        )
        results['onnx_inference_ms'] = onnx_benchmark['mean_ms']
        results['onnx_throughput_fps'] = onnx_benchmark['throughput_fps']
    
    # Quantize
    if quantize:
        quantized_path = output_dir / 'cropshield_quantized.onnx'
        
        quant_success = quantize_onnx_dynamic(
            onnx_path=str(onnx_path),
            output_path=str(quantized_path),
            weight_type='int8',
            verbose=verbose
        )
        
        if quant_success:
            results['quantized_path'] = str(quantized_path)
            results['quantized_size_mb'] = quantized_path.stat().st_size / (1024 * 1024)
            
            # Compare models
            comparison = compare_models(
                original_onnx=str(onnx_path),
                quantized_onnx=str(quantized_path),
                num_samples=10,
                verbose=verbose
            )
            results.update(comparison)
    
    # Save export info
    export_info = {
        'model_path': model_path,
        'model_type': model_type,
        'num_classes': num_classes,
        'class_names': class_names,
        'onnx_path': str(onnx_path),
        'quantized_path': str(results.get('quantized_path', '')),
        'results': results
    }
    
    info_path = output_dir / 'export_info.json'
    with open(info_path, 'w') as f:
        json.dump(export_info, indent=2, fp=f)
    
    if verbose:
        print(f"\n✅ Export info saved to: {info_path}")
    
    return results


# ============================================================
# EXAMPLE USAGE
# ============================================================

def example_export_custom_cnn():
    """Example: Export Custom CNN model."""
    
    model_path = 'models/custom_best.pth'
    
    results = export_full_pipeline(
        model_path=model_path,
        model_type='custom',
        num_classes=22,
        output_dir='models',
        quantize=True,
        verify=True,
        benchmark=True
    )
    
    return results


def example_export_efficientnet():
    """Example: Export EfficientNet-B0 model."""
    
    model_path = 'models/efficientnet_b0_best.pth'
    
    results = export_full_pipeline(
        model_path=model_path,
        model_type='efficientnet_b0',
        num_classes=22,
        output_dir='models',
        quantize=True,
        verify=True,
        benchmark=True
    )
    
    return results


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Export PyTorch model to ONNX')
    parser.add_argument('--model_path', type=str, required=True,
                       help='Path to trained PyTorch model (.pth)')
    parser.add_argument('--model_type', type=str, required=True,
                       choices=['custom', 'efficientnet_b0'],
                       help='Model architecture')
    parser.add_argument('--num_classes', type=int, default=22,
                       help='Number of classes (default: 22)')
    parser.add_argument('--output_dir', type=str, default='models',
                       help='Output directory (default: models)')
    parser.add_argument('--quantize', action='store_true',
                       help='Apply dynamic quantization')
    parser.add_argument('--no_verify', action='store_true',
                       help='Skip inference verification')
    parser.add_argument('--no_benchmark', action='store_true',
                       help='Skip benchmarking')
    
    args = parser.parse_args()
    
    print(f"\n🚀 CropShield AI - ONNX Export\n")
    
    results = export_full_pipeline(
        model_path=args.model_path,
        model_type=args.model_type,
        num_classes=args.num_classes,
        output_dir=args.output_dir,
        quantize=args.quantize,
        verify=not args.no_verify,
        benchmark=not args.no_benchmark,
        verbose=True
    )
    
    if results:
        print(f"\n✅ Export complete!")
        print(f"   ONNX model: {results.get('onnx_path')}")
        if 'quantized_path' in results:
            print(f"   Quantized model: {results.get('quantized_path')}")
    else:
        print(f"\n❌ Export failed")
