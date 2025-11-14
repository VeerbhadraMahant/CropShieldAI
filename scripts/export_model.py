"""
CropShield AI - Model Export Script
====================================

Exports trained PyTorch models to TorchScript and ONNX formats for fast, portable inference.

Features:
- TorchScript export: Optimized PyTorch format (2-5x faster)
- ONNX export: Cross-platform format (runs on ONNX Runtime)
- Dynamic batch dimension support
- Parity testing across all formats
- Detailed export validation
- Model size and performance comparison

Formats:
    PyTorch (.pth)     : Full PyTorch model (requires torch)
    TorchScript (.pt)  : Traced/scripted model (lighter PyTorch dependency)
    ONNX (.onnx)       : Open format (no PyTorch needed, fastest inference)

Usage:
    # Basic export (all formats)
    python scripts/export_model.py --model models/cropshield_cnn.pth
    
    # Export specific formats
    python scripts/export_model.py --model models/cropshield_cnn.pth --formats torchscript onnx
    
    # Custom output directory
    python scripts/export_model.py --model models/cropshield_cnn.pth --output exported_models/
    
    # With custom input shape
    python scripts/export_model.py --model models/cropshield_cnn.pth --input_shape 1 3 256 256
    
    # Skip parity testing (faster)
    python scripts/export_model.py --model models/cropshield_cnn.pth --skip_test

Output Files:
    cropshield_cnn_scripted.pt  : TorchScript traced model
    cropshield_cnn.onnx         : ONNX model with dynamic batch axis

Compatibility:
    - TorchScript: Compatible with PyTorch inference and GradCAM
    - ONNX: Pure inference only (no gradient hooks for GradCAM)
    - Both formats maintain identical prediction outputs
"""

import torch
import torch.nn as nn
import torch.onnx
from pathlib import Path
import argparse
import sys
import time
import json
import warnings
from typing import Tuple, Optional, List, Dict
import numpy as np

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from model_custom_cnn import CropShieldCNN

# Suppress warnings
warnings.filterwarnings('ignore')


def load_pytorch_model(
    model_path: str,
    num_classes: int = 22,
    device: str = 'cpu'
) -> nn.Module:
    """
    Load trained PyTorch model from .pth file.
    
    Args:
        model_path: Path to .pth model file
        num_classes: Number of output classes (default: 22)
        device: Device to load model on ('cpu' or 'cuda')
    
    Returns:
        Loaded model in evaluation mode
    
    Raises:
        FileNotFoundError: If model file doesn't exist
        RuntimeError: If model loading fails
    """
    model_path = Path(model_path)
    
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")
    
    print(f"📂 Loading PyTorch model from: {model_path}")
    
    # Initialize model architecture
    model = CropShieldCNN(num_classes=num_classes)
    
    # Load state dict
    try:
        checkpoint = torch.load(model_path, map_location=device)
        
        # Handle different checkpoint formats
        if isinstance(checkpoint, dict):
            if 'model_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['model_state_dict'])
                print(f"   ✓ Loaded from checkpoint (epoch {checkpoint.get('epoch', 'unknown')})")
            elif 'state_dict' in checkpoint:
                model.load_state_dict(checkpoint['state_dict'])
                print(f"   ✓ Loaded from checkpoint with state_dict")
            else:
                model.load_state_dict(checkpoint)
                print(f"   ✓ Loaded state dict")
        else:
            model.load_state_dict(checkpoint)
            print(f"   ✓ Loaded model weights")
    
    except Exception as e:
        raise RuntimeError(f"Failed to load model: {e}")
    
    # Set to evaluation mode
    model.eval()
    model.to(device)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"   ✓ Model loaded successfully")
    print(f"   ✓ Total parameters: {total_params:,}")
    print(f"   ✓ Trainable parameters: {trainable_params:,}")
    print(f"   ✓ Device: {device}")
    
    return model


def export_to_torchscript(
    model: nn.Module,
    output_path: str,
    input_shape: Tuple[int, int, int, int] = (1, 3, 224, 224),
    device: str = 'cpu'
) -> Path:
    """
    Export model to TorchScript format using torch.jit.trace().
    
    TorchScript Benefits:
    - 2-5x faster inference than PyTorch
    - Reduced PyTorch dependency (no Python runtime needed)
    - Compatible with C++ deployment
    - Preserves model structure for GradCAM hooks
    
    Args:
        model: PyTorch model in eval mode
        output_path: Output file path (will add .pt extension if missing)
        input_shape: Input tensor shape (B, C, H, W)
        device: Device for tracing
    
    Returns:
        Path to exported .pt file
    
    Raises:
        RuntimeError: If export fails
    """
    output_path = Path(output_path)
    if output_path.suffix != '.pt':
        output_path = output_path.with_suffix('.pt')
    
    print(f"\n🔄 Exporting to TorchScript...")
    print(f"   Output: {output_path}")
    
    try:
        # Create example input for tracing
        example_input = torch.randn(input_shape).to(device)
        
        # Trace the model
        print(f"   ⏳ Tracing model with input shape {input_shape}...")
        start_time = time.time()
        
        with torch.no_grad():
            traced_model = torch.jit.trace(model, example_input)
        
        trace_time = time.time() - start_time
        
        # Verify traced model works
        print(f"   ⏳ Verifying traced model...")
        traced_output = traced_model(example_input)
        
        # Save traced model
        print(f"   ⏳ Saving TorchScript model...")
        traced_model.save(str(output_path))
        
        # Get file size
        file_size_mb = output_path.stat().st_size / (1024 * 1024)
        
        print(f"   ✅ TorchScript export successful!")
        print(f"   ✓ Trace time: {trace_time:.3f}s")
        print(f"   ✓ File size: {file_size_mb:.2f} MB")
        print(f"   ✓ Output shape: {tuple(traced_output.shape)}")
        
        return output_path
    
    except Exception as e:
        raise RuntimeError(f"TorchScript export failed: {e}")


def export_to_onnx(
    model: nn.Module,
    output_path: str,
    input_shape: Tuple[int, int, int, int] = (1, 3, 224, 224),
    device: str = 'cpu',
    opset_version: int = 14,
    dynamic_axes: bool = True
) -> Path:
    """
    Export model to ONNX format.
    
    ONNX Benefits:
    - Cross-platform inference (CPU, GPU, mobile, edge devices)
    - ONNX Runtime is faster than PyTorch (2-10x speedup)
    - No PyTorch dependency needed
    - Smaller file size
    - Dynamic batch dimension support
    
    Args:
        model: PyTorch model in eval mode
        output_path: Output file path (will add .onnx extension if missing)
        input_shape: Input tensor shape (B, C, H, W)
        device: Device for export
        opset_version: ONNX opset version (14 recommended for compatibility)
        dynamic_axes: Enable dynamic batch dimension
    
    Returns:
        Path to exported .onnx file
    
    Raises:
        RuntimeError: If export fails
    
    Note:
        ONNX models don't support gradient hooks, so GradCAM won't work.
        Use TorchScript for GradCAM compatibility.
    """
    output_path = Path(output_path)
    if output_path.suffix != '.onnx':
        output_path = output_path.with_suffix('.onnx')
    
    print(f"\n🔄 Exporting to ONNX...")
    print(f"   Output: {output_path}")
    print(f"   Opset version: {opset_version}")
    print(f"   Dynamic batch: {dynamic_axes}")
    
    try:
        # Create example input
        example_input = torch.randn(input_shape).to(device)
        
        # Configure dynamic axes
        if dynamic_axes:
            dynamic_ax = {
                'input': {0: 'batch_size'},  # Dynamic batch dimension
                'output': {0: 'batch_size'}
            }
            print(f"   ✓ Dynamic axes configured: batch_size variable")
        else:
            dynamic_ax = None
        
        # Export to ONNX
        print(f"   ⏳ Exporting model with input shape {input_shape}...")
        start_time = time.time()
        
        torch.onnx.export(
            model,                      # Model to export
            example_input,              # Example input tensor
            str(output_path),          # Output path
            export_params=True,         # Store trained parameters
            opset_version=opset_version, # ONNX version
            do_constant_folding=True,   # Optimize constants
            input_names=['input'],      # Input name
            output_names=['output'],    # Output name
            dynamic_axes=dynamic_ax,    # Dynamic axes
            verbose=False               # Suppress verbose output
        )
        
        export_time = time.time() - start_time
        
        # Get file size
        file_size_mb = output_path.stat().st_size / (1024 * 1024)
        
        print(f"   ✅ ONNX export successful!")
        print(f"   ✓ Export time: {export_time:.3f}s")
        print(f"   ✓ File size: {file_size_mb:.2f} MB")
        
        # Verify ONNX model (optional)
        try:
            import onnx
            onnx_model = onnx.load(str(output_path))
            onnx.checker.check_model(onnx_model)
            print(f"   ✓ ONNX model validation: PASSED")
        except ImportError:
            print(f"   ⚠️  Install 'onnx' package to validate exported model")
        except Exception as e:
            print(f"   ⚠️  ONNX validation warning: {e}")
        
        return output_path
    
    except Exception as e:
        raise RuntimeError(f"ONNX export failed: {e}")


def test_model_parity(
    pytorch_model: nn.Module,
    torchscript_path: Optional[Path] = None,
    onnx_path: Optional[Path] = None,
    input_shape: Tuple[int, int, int, int] = (1, 3, 224, 224),
    device: str = 'cpu',
    num_samples: int = 5,
    tolerance: float = 1e-5
) -> Dict[str, Dict[str, float]]:
    """
    Test output parity between PyTorch, TorchScript, and ONNX models.
    
    Validates that all exported models produce identical outputs (within tolerance)
    for the same input. This ensures export correctness.
    
    Args:
        pytorch_model: Original PyTorch model
        torchscript_path: Path to TorchScript .pt file (optional)
        onnx_path: Path to ONNX .onnx file (optional)
        input_shape: Input tensor shape for testing
        device: Device for testing
        num_samples: Number of random samples to test
        tolerance: Maximum allowed difference between outputs
    
    Returns:
        Dictionary with parity test results and statistics
    
    Example:
        {
            'torchscript': {'max_diff': 1.23e-6, 'mean_diff': 5.67e-7, 'passed': True},
            'onnx': {'max_diff': 2.34e-6, 'mean_diff': 8.90e-7, 'passed': True}
        }
    """
    print(f"\n🧪 Testing model parity...")
    print(f"   Samples: {num_samples}")
    print(f"   Tolerance: {tolerance:.2e}")
    
    results = {}
    
    # Load TorchScript model if provided
    if torchscript_path and torchscript_path.exists():
        print(f"\n   📊 Testing TorchScript vs PyTorch...")
        try:
            traced_model = torch.jit.load(str(torchscript_path), map_location=device)
            traced_model.eval()
            
            max_diffs = []
            mean_diffs = []
            
            with torch.no_grad():
                for i in range(num_samples):
                    # Random input
                    test_input = torch.randn(input_shape).to(device)
                    
                    # Get outputs
                    pytorch_output = pytorch_model(test_input)
                    torchscript_output = traced_model(test_input)
                    
                    # Compute differences
                    diff = torch.abs(pytorch_output - torchscript_output)
                    max_diff = diff.max().item()
                    mean_diff = diff.mean().item()
                    
                    max_diffs.append(max_diff)
                    mean_diffs.append(mean_diff)
            
            max_diff_overall = max(max_diffs)
            mean_diff_overall = np.mean(mean_diffs)
            passed = max_diff_overall < tolerance
            
            results['torchscript'] = {
                'max_diff': max_diff_overall,
                'mean_diff': mean_diff_overall,
                'passed': passed
            }
            
            status = "✅ PASSED" if passed else "❌ FAILED"
            print(f"      {status}")
            print(f"      Max difference:  {max_diff_overall:.6e}")
            print(f"      Mean difference: {mean_diff_overall:.6e}")
            
        except Exception as e:
            print(f"      ❌ TorchScript test failed: {e}")
            results['torchscript'] = {'error': str(e)}
    
    # Load ONNX model if provided
    if onnx_path and onnx_path.exists():
        print(f"\n   📊 Testing ONNX vs PyTorch...")
        try:
            import onnxruntime as ort
            
            # Create ONNX Runtime session
            ort_session = ort.InferenceSession(
                str(onnx_path),
                providers=['CPUExecutionProvider']
            )
            
            max_diffs = []
            mean_diffs = []
            
            with torch.no_grad():
                for i in range(num_samples):
                    # Random input
                    test_input = torch.randn(input_shape).to(device)
                    
                    # PyTorch output
                    pytorch_output = pytorch_model(test_input).cpu().numpy()
                    
                    # ONNX output
                    onnx_input = {ort_session.get_inputs()[0].name: test_input.cpu().numpy()}
                    onnx_output = ort_session.run(None, onnx_input)[0]
                    
                    # Compute differences
                    diff = np.abs(pytorch_output - onnx_output)
                    max_diff = diff.max()
                    mean_diff = diff.mean()
                    
                    max_diffs.append(max_diff)
                    mean_diffs.append(mean_diff)
            
            max_diff_overall = max(max_diffs)
            mean_diff_overall = np.mean(mean_diffs)
            passed = max_diff_overall < tolerance
            
            results['onnx'] = {
                'max_diff': max_diff_overall,
                'mean_diff': mean_diff_overall,
                'passed': passed
            }
            
            status = "✅ PASSED" if passed else "❌ FAILED"
            print(f"      {status}")
            print(f"      Max difference:  {max_diff_overall:.6e}")
            print(f"      Mean difference: {mean_diff_overall:.6e}")
            
        except ImportError:
            print(f"      ⚠️  Install 'onnxruntime' to test ONNX model")
            print(f"         pip install onnxruntime")
            results['onnx'] = {'error': 'onnxruntime not installed'}
        except Exception as e:
            print(f"      ❌ ONNX test failed: {e}")
            results['onnx'] = {'error': str(e)}
    
    return results


def compare_inference_speed(
    pytorch_model: nn.Module,
    torchscript_path: Optional[Path] = None,
    onnx_path: Optional[Path] = None,
    input_shape: Tuple[int, int, int, int] = (1, 3, 224, 224),
    device: str = 'cpu',
    num_iterations: int = 100,
    warmup: int = 10
) -> Dict[str, float]:
    """
    Compare inference speed across model formats.
    
    Args:
        pytorch_model: Original PyTorch model
        torchscript_path: Path to TorchScript model
        onnx_path: Path to ONNX model
        input_shape: Input shape for testing
        device: Device for testing
        num_iterations: Number of inference iterations
        warmup: Number of warmup iterations
    
    Returns:
        Dictionary with average inference times (ms) for each format
    """
    print(f"\n⚡ Comparing inference speed...")
    print(f"   Iterations: {num_iterations} (+ {warmup} warmup)")
    print(f"   Device: {device}")
    
    results = {}
    test_input = torch.randn(input_shape).to(device)
    
    # PyTorch speed
    print(f"\n   🔥 PyTorch inference...")
    pytorch_model.eval()
    
    # Warmup
    with torch.no_grad():
        for _ in range(warmup):
            _ = pytorch_model(test_input)
    
    # Benchmark
    start_time = time.time()
    with torch.no_grad():
        for _ in range(num_iterations):
            _ = pytorch_model(test_input)
    
    if device == 'cuda':
        torch.cuda.synchronize()
    
    pytorch_time = (time.time() - start_time) / num_iterations * 1000
    results['pytorch'] = pytorch_time
    print(f"      ✓ {pytorch_time:.3f} ms/image")
    
    # TorchScript speed
    if torchscript_path and torchscript_path.exists():
        print(f"\n   🔥 TorchScript inference...")
        try:
            traced_model = torch.jit.load(str(torchscript_path), map_location=device)
            traced_model.eval()
            
            # Warmup
            with torch.no_grad():
                for _ in range(warmup):
                    _ = traced_model(test_input)
            
            # Benchmark
            start_time = time.time()
            with torch.no_grad():
                for _ in range(num_iterations):
                    _ = traced_model(test_input)
            
            if device == 'cuda':
                torch.cuda.synchronize()
            
            torchscript_time = (time.time() - start_time) / num_iterations * 1000
            results['torchscript'] = torchscript_time
            speedup = pytorch_time / torchscript_time
            print(f"      ✓ {torchscript_time:.3f} ms/image ({speedup:.2f}x speedup)")
        
        except Exception as e:
            print(f"      ❌ TorchScript benchmark failed: {e}")
    
    # ONNX speed
    if onnx_path and onnx_path.exists():
        print(f"\n   🔥 ONNX Runtime inference...")
        try:
            import onnxruntime as ort
            
            ort_session = ort.InferenceSession(
                str(onnx_path),
                providers=['CPUExecutionProvider']
            )
            
            test_input_np = test_input.cpu().numpy()
            input_name = ort_session.get_inputs()[0].name
            
            # Warmup
            for _ in range(warmup):
                _ = ort_session.run(None, {input_name: test_input_np})
            
            # Benchmark
            start_time = time.time()
            for _ in range(num_iterations):
                _ = ort_session.run(None, {input_name: test_input_np})
            
            onnx_time = (time.time() - start_time) / num_iterations * 1000
            results['onnx'] = onnx_time
            speedup = pytorch_time / onnx_time
            print(f"      ✓ {onnx_time:.3f} ms/image ({speedup:.2f}x speedup)")
        
        except ImportError:
            print(f"      ⚠️  Install 'onnxruntime' to benchmark ONNX")
        except Exception as e:
            print(f"      ❌ ONNX benchmark failed: {e}")
    
    return results


def print_export_summary(
    pytorch_path: Path,
    torchscript_path: Optional[Path] = None,
    onnx_path: Optional[Path] = None,
    parity_results: Optional[Dict] = None,
    speed_results: Optional[Dict] = None
):
    """
    Print comprehensive export summary.
    
    Args:
        pytorch_path: Path to original PyTorch model
        torchscript_path: Path to TorchScript export
        onnx_path: Path to ONNX export
        parity_results: Parity test results
        speed_results: Speed comparison results
    """
    print(f"\n" + "="*70)
    print(f"📦 EXPORT SUMMARY")
    print(f"="*70)
    
    # File sizes
    print(f"\n📊 Model Files:")
    print(f"   PyTorch (.pth):     {pytorch_path.stat().st_size / (1024*1024):.2f} MB")
    
    if torchscript_path and torchscript_path.exists():
        print(f"   TorchScript (.pt):  {torchscript_path.stat().st_size / (1024*1024):.2f} MB")
    
    if onnx_path and onnx_path.exists():
        print(f"   ONNX (.onnx):       {onnx_path.stat().st_size / (1024*1024):.2f} MB")
    
    # Parity results
    if parity_results:
        print(f"\n✅ Parity Tests:")
        for format_name, result in parity_results.items():
            if 'error' in result:
                print(f"   {format_name.upper()}: ❌ {result['error']}")
            else:
                status = "✅ PASSED" if result['passed'] else "❌ FAILED"
                print(f"   {format_name.upper()}: {status} (max diff: {result['max_diff']:.6e})")
    
    # Speed results
    if speed_results:
        print(f"\n⚡ Inference Speed:")
        for format_name, time_ms in speed_results.items():
            print(f"   {format_name.upper()}: {time_ms:.3f} ms/image")
    
    # Usage examples
    print(f"\n💡 Usage Examples:")
    print(f"\n   PyTorch Inference:")
    print(f"   -----------------")
    print(f"   from predict import load_model_once, predict_disease")
    print(f"   model, classes, device = load_model_once('{pytorch_path.name}')")
    print(f"   predictions = predict_disease('image.jpg', model, classes, device)")
    
    if torchscript_path and torchscript_path.exists():
        print(f"\n   TorchScript Inference:")
        print(f"   ---------------------")
        print(f"   model = torch.jit.load('{torchscript_path.name}')")
        print(f"   output = model(input_tensor)")
    
    if onnx_path and onnx_path.exists():
        print(f"\n   ONNX Runtime Inference:")
        print(f"   ----------------------")
        print(f"   import onnxruntime as ort")
        print(f"   session = ort.InferenceSession('{onnx_path.name}')")
        print(f"   output = session.run(None, {{'input': input_array}})[0]")
    
    print(f"\n" + "="*70)
    print(f"✅ Export complete!")
    print(f"="*70 + "\n")


def main():
    parser = argparse.ArgumentParser(
        description='Export CropShield AI models to TorchScript and ONNX',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Export all formats
  python scripts/export_model.py --model models/cropshield_cnn.pth
  
  # Export specific formats
  python scripts/export_model.py --model models/cropshield_cnn.pth --formats torchscript
  
  # Custom output directory
  python scripts/export_model.py --model models/cropshield_cnn.pth --output exported/
  
  # Skip parity testing (faster)
  python scripts/export_model.py --model models/cropshield_cnn.pth --skip_test

For more information, visit: https://github.com/yourusername/CropShieldAI
        """
    )
    
    parser.add_argument(
        '--model',
        type=str,
        required=True,
        help='Path to trained PyTorch model (.pth file)'
    )
    
    parser.add_argument(
        '--output',
        type=str,
        default=None,
        help='Output directory for exported models (default: same as input model)'
    )
    
    parser.add_argument(
        '--formats',
        nargs='+',
        choices=['torchscript', 'onnx', 'all'],
        default=['all'],
        help='Export formats (default: all)'
    )
    
    parser.add_argument(
        '--input_shape',
        nargs=4,
        type=int,
        default=[1, 3, 224, 224],
        metavar=('B', 'C', 'H', 'W'),
        help='Input shape: batch channels height width (default: 1 3 224 224)'
    )
    
    parser.add_argument(
        '--num_classes',
        type=int,
        default=22,
        help='Number of output classes (default: 22)'
    )
    
    parser.add_argument(
        '--device',
        type=str,
        default='cpu',
        choices=['cpu', 'cuda'],
        help='Device for export and testing (default: cpu)'
    )
    
    parser.add_argument(
        '--skip_test',
        action='store_true',
        help='Skip parity testing (faster export)'
    )
    
    parser.add_argument(
        '--skip_benchmark',
        action='store_true',
        help='Skip inference speed benchmark'
    )
    
    parser.add_argument(
        '--opset',
        type=int,
        default=14,
        help='ONNX opset version (default: 14)'
    )
    
    args = parser.parse_args()
    
    # Validate model path
    model_path = Path(args.model)
    if not model_path.exists():
        print(f"❌ Error: Model file not found: {model_path}")
        sys.exit(1)
    
    # Determine output directory
    if args.output:
        output_dir = Path(args.output)
    else:
        output_dir = model_path.parent
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Determine formats to export
    formats = args.formats
    if 'all' in formats:
        formats = ['torchscript', 'onnx']
    
    # Setup
    input_shape = tuple(args.input_shape)
    device = args.device
    
    if device == 'cuda' and not torch.cuda.is_available():
        print(f"⚠️  CUDA not available, using CPU")
        device = 'cpu'
    
    print(f"\n{'='*70}")
    print(f"🚀 CropShield AI Model Export")
    print(f"{'='*70}")
    print(f"   Model: {model_path}")
    print(f"   Output: {output_dir}")
    print(f"   Formats: {', '.join(formats)}")
    print(f"   Input shape: {input_shape}")
    print(f"   Device: {device}")
    print(f"{'='*70}\n")
    
    try:
        # Load PyTorch model
        pytorch_model = load_pytorch_model(
            model_path,
            num_classes=args.num_classes,
            device=device
        )
        
        # Export paths
        model_base_name = model_path.stem
        torchscript_path = None
        onnx_path = None
        
        # Export to TorchScript
        if 'torchscript' in formats:
            torchscript_path = export_to_torchscript(
                model=pytorch_model,
                output_path=output_dir / f"{model_base_name}_scripted.pt",
                input_shape=input_shape,
                device=device
            )
        
        # Export to ONNX
        if 'onnx' in formats:
            onnx_path = export_to_onnx(
                model=pytorch_model,
                output_path=output_dir / f"{model_base_name}.onnx",
                input_shape=input_shape,
                device=device,
                opset_version=args.opset,
                dynamic_axes=True
            )
        
        # Test parity
        parity_results = None
        if not args.skip_test:
            parity_results = test_model_parity(
                pytorch_model=pytorch_model,
                torchscript_path=torchscript_path,
                onnx_path=onnx_path,
                input_shape=input_shape,
                device=device
            )
        
        # Benchmark speed
        speed_results = None
        if not args.skip_benchmark:
            speed_results = compare_inference_speed(
                pytorch_model=pytorch_model,
                torchscript_path=torchscript_path,
                onnx_path=onnx_path,
                input_shape=input_shape,
                device=device
            )
        
        # Print summary
        print_export_summary(
            pytorch_path=model_path,
            torchscript_path=torchscript_path,
            onnx_path=onnx_path,
            parity_results=parity_results,
            speed_results=speed_results
        )
        
        print(f"✅ All exports completed successfully!\n")
        
    except Exception as e:
        print(f"\n❌ Export failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
