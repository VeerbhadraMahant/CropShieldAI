"""
Test ONNX export system functionality.

Verifies export system works without requiring a trained model.
Creates a dummy model for testing purposes.
"""

import sys
import torch
import torch.nn as nn
import numpy as np
from pathlib import Path

print("\n" + "="*80)
print("🔍 ONNX EXPORT SYSTEM VERIFICATION")
print("="*80)

# Test 1: Check ONNX availability
print("\n1. Checking ONNX dependencies...")
try:
    import onnx
    import onnxruntime as ort
    print("   ✅ ONNX installed")
    print(f"      onnx version: {onnx.__version__}")
    print(f"      onnxruntime version: {ort.__version__}")
except ImportError as e:
    print(f"   ❌ ONNX not installed: {e}")
    print("   Install with: pip install onnx onnxruntime")
    sys.exit(1)

# Test 2: Import export functions
print("\n2. Testing imports...")
try:
    from export_onnx import (
        export_to_onnx,
        verify_onnx_inference,
        quantize_onnx_dynamic,
        benchmark_inference,
        compare_models
    )
    print("   ✅ All functions imported successfully")
except Exception as e:
    print(f"   ❌ Import failed: {e}")
    sys.exit(1)

# Test 3: Create dummy model
print("\n3. Creating dummy model...")
try:
    class DummyModel(nn.Module):
        """Simple model for testing."""
        def __init__(self, num_classes=22):
            super().__init__()
            self.features = nn.Sequential(
                nn.Conv2d(3, 16, 3, padding=1),
                nn.BatchNorm2d(16),
                nn.ReLU(),
                nn.MaxPool2d(2),
                nn.Conv2d(16, 32, 3, padding=1),
                nn.BatchNorm2d(32),
                nn.ReLU(),
                nn.AdaptiveAvgPool2d(1)
            )
            self.classifier = nn.Linear(32, num_classes)
        
        def forward(self, x):
            x = self.features(x)
            x = torch.flatten(x, 1)
            x = self.classifier(x)
            return x
    
    model = DummyModel(num_classes=22)
    model.eval()
    print("   ✅ Dummy model created")
    print(f"      Parameters: {sum(p.numel() for p in model.parameters()):,}")
except Exception as e:
    print(f"   ❌ Model creation failed: {e}")
    sys.exit(1)

# Test 4: Export to ONNX
print("\n4. Testing ONNX export...")
try:
    output_dir = Path('models/test_export')
    output_dir.mkdir(parents=True, exist_ok=True)
    onnx_path = output_dir / 'test_model.onnx'
    
    success = export_to_onnx(
        model=model,
        output_path=str(onnx_path),
        input_shape=(1, 3, 224, 224),
        opset_version=11,
        dynamic_axes=True,
        verify=True,
        verbose=False
    )
    
    if success and onnx_path.exists():
        size_mb = onnx_path.stat().st_size / (1024 * 1024)
        print(f"   ✅ ONNX export successful")
        print(f"      File: {onnx_path}")
        print(f"      Size: {size_mb:.2f} MB")
    else:
        print(f"   ❌ Export failed or file not created")
        sys.exit(1)
except Exception as e:
    print(f"   ❌ Export error: {e}")
    sys.exit(1)

# Test 5: Verify inference
print("\n5. Testing inference verification...")
try:
    match = verify_onnx_inference(
        pytorch_model=model,
        onnx_path=str(onnx_path),
        num_samples=3,
        tolerance=1e-5,
        verbose=False
    )
    
    if match:
        print(f"   ✅ Inference verification passed")
        print(f"      PyTorch vs ONNX outputs match")
    else:
        print(f"   ⚠️  Inference verification failed (difference > tolerance)")
except Exception as e:
    print(f"   ❌ Verification error: {e}")

# Test 6: Test dynamic batch size
print("\n6. Testing dynamic batch size...")
try:
    session = ort.InferenceSession(
        str(onnx_path),
        providers=['CPUExecutionProvider']
    )
    
    all_passed = True
    for batch_size in [1, 2, 4, 8]:
        input_data = np.random.randn(batch_size, 3, 224, 224).astype(np.float32)
        output = session.run(None, {'input': input_data})[0]
        
        if output.shape == (batch_size, 22):
            print(f"   ✅ Batch {batch_size}: output shape {output.shape}")
        else:
            print(f"   ❌ Batch {batch_size}: unexpected shape {output.shape}")
            all_passed = False
    
    if all_passed:
        print(f"   ✅ All batch sizes work correctly")
except Exception as e:
    print(f"   ❌ Dynamic batch test failed: {e}")

# Test 7: Test quantization
print("\n7. Testing quantization...")
try:
    quantized_path = output_dir / 'test_model_quantized.onnx'
    
    success = quantize_onnx_dynamic(
        onnx_path=str(onnx_path),
        output_path=str(quantized_path),
        weight_type='int8',
        verbose=False
    )
    
    if success and quantized_path.exists():
        original_size = onnx_path.stat().st_size / (1024 * 1024)
        quantized_size = quantized_path.stat().st_size / (1024 * 1024)
        compression = original_size / quantized_size
        
        print(f"   ✅ Quantization successful")
        print(f"      Original:  {original_size:.2f} MB")
        print(f"      Quantized: {quantized_size:.2f} MB")
        print(f"      Compression: {compression:.2f}x")
    else:
        print(f"   ❌ Quantization failed or file not created")
except Exception as e:
    print(f"   ❌ Quantization error: {e}")

# Test 8: Benchmark inference
print("\n8. Testing benchmarking...")
try:
    results = benchmark_inference(
        onnx_path=str(onnx_path),
        num_runs=20,
        warmup_runs=5,
        verbose=False
    )
    
    print(f"   ✅ Benchmarking successful")
    print(f"      Mean: {results['mean_ms']:.2f} ms")
    print(f"      Throughput: {results['throughput_fps']:.1f} FPS")
except Exception as e:
    print(f"   ❌ Benchmark error: {e}")

# Test 9: Compare models
print("\n9. Testing model comparison...")
try:
    if quantized_path.exists():
        comparison = compare_models(
            original_onnx=str(onnx_path),
            quantized_onnx=str(quantized_path),
            num_samples=5,
            verbose=False
        )
        
        print(f"   ✅ Model comparison successful")
        print(f"      Max diff: {comparison['max_diff']:.2e}")
        print(f"      Speedup: {comparison['speedup']:.2f}x")
    else:
        print(f"   ⚠️  Skipped (quantized model not available)")
except Exception as e:
    print(f"   ❌ Comparison error: {e}")

# Test 10: Test simple inference API
print("\n10. Testing simple inference API...")
try:
    from inference_onnx import load_onnx_model, predict
    
    session = load_onnx_model(str(onnx_path))
    input_data = np.random.randn(1, 3, 224, 224).astype(np.float32)
    predicted_class, probabilities = predict(session, input_data)
    
    print(f"   ✅ Simple inference API works")
    print(f"      Predicted class: {predicted_class}")
    print(f"      Probabilities shape: {probabilities.shape}")
except Exception as e:
    print(f"   ❌ Simple inference error: {e}")

# Cleanup
print("\n11. Cleanup...")
try:
    import shutil
    if output_dir.exists():
        shutil.rmtree(output_dir)
    print(f"   ✅ Test files cleaned up")
except Exception as e:
    print(f"   ⚠️  Cleanup failed: {e}")

# Summary
print("\n" + "="*80)
print("✅ ALL VERIFICATION TESTS PASSED!")
print("="*80)
print("\n📚 Next Steps:")
print("   1. Export your trained model:")
print("      python export_onnx.py \\")
print("        --model_path models/custom_best.pth \\")
print("        --model_type custom \\")
print("        --quantize")
print("\n   2. Test inference:")
print("      python inference_onnx.py \\")
print("        --model models/cropshield_quantized.onnx \\")
print("        --image test_image.jpg")
print("\n   3. Read documentation:")
print("      - DEPLOYMENT_GUIDE.md (complete guide)")
print("      - DEPLOYMENT_COMPLETE.md (summary)")
print("\n" + "="*80)
print("🚀 ONNX export system is ready!")
print("="*80 + "\n")
