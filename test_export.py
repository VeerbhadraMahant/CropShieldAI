"""
Test Export Functionality
=========================

Test script to verify model export to TorchScript and ONNX works correctly.

This script creates a dummy model, exports it to all formats, and verifies:
1. Export completes successfully
2. Output parity between formats
3. Model can be loaded and used for inference
4. Dynamic batch dimension works (ONNX)

Usage:
    python test_export.py
"""

import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
import sys
import tempfile
import shutil

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from model_custom_cnn import CropShieldCNN


def create_dummy_model(num_classes: int = 22) -> nn.Module:
    """Create and initialize a dummy model for testing."""
    print("🔨 Creating dummy model...")
    
    model = CropShieldCNN(num_classes=num_classes)
    model.eval()
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"   ✓ Model created with {total_params:,} parameters")
    
    return model


def save_dummy_checkpoint(model: nn.Module, path: Path):
    """Save model as PyTorch checkpoint."""
    print(f"💾 Saving dummy checkpoint to {path}")
    
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'epoch': 0,
        'num_classes': 22
    }
    
    torch.save(checkpoint, path)
    print(f"   ✓ Checkpoint saved ({path.stat().st_size / (1024*1024):.2f} MB)")


def test_torchscript_export(model: nn.Module, output_path: Path) -> bool:
    """Test TorchScript export."""
    print("\n🔄 Testing TorchScript export...")
    
    try:
        # Create example input
        example_input = torch.randn(1, 3, 224, 224)
        
        # Trace model
        print("   ⏳ Tracing model...")
        traced_model = torch.jit.trace(model, example_input)
        
        # Save
        print(f"   ⏳ Saving to {output_path}...")
        traced_model.save(str(output_path))
        
        # Verify file exists
        if not output_path.exists():
            print("   ❌ Export failed: File not created")
            return False
        
        # Test loading
        print("   ⏳ Testing model loading...")
        loaded_model = torch.jit.load(str(output_path))
        
        # Test inference
        print("   ⏳ Testing inference...")
        with torch.no_grad():
            output = loaded_model(example_input)
        
        if output.shape != (1, 22):
            print(f"   ❌ Wrong output shape: {output.shape}, expected (1, 22)")
            return False
        
        file_size_mb = output_path.stat().st_size / (1024 * 1024)
        print(f"   ✅ TorchScript export successful!")
        print(f"   ✓ File size: {file_size_mb:.2f} MB")
        print(f"   ✓ Output shape: {tuple(output.shape)}")
        
        return True
    
    except Exception as e:
        print(f"   ❌ TorchScript export failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_onnx_export(model: nn.Module, output_path: Path) -> bool:
    """Test ONNX export."""
    print("\n🔄 Testing ONNX export...")
    
    try:
        # Create example input
        example_input = torch.randn(1, 3, 224, 224)
        
        # Export to ONNX
        print("   ⏳ Exporting to ONNX...")
        torch.onnx.export(
            model,
            example_input,
            str(output_path),
            export_params=True,
            opset_version=14,
            do_constant_folding=True,
            input_names=['input'],
            output_names=['output'],
            dynamic_axes={
                'input': {0: 'batch_size'},
                'output': {0: 'batch_size'}
            }
        )
        
        # Verify file exists
        if not output_path.exists():
            print("   ❌ Export failed: File not created")
            return False
        
        file_size_mb = output_path.stat().st_size / (1024 * 1024)
        print(f"   ✅ ONNX export successful!")
        print(f"   ✓ File size: {file_size_mb:.2f} MB")
        
        # Test with ONNX Runtime if available
        try:
            import onnxruntime as ort
            
            print("   ⏳ Testing ONNX Runtime inference...")
            session = ort.InferenceSession(str(output_path))
            
            # Test single image
            input_array = example_input.numpy()
            output = session.run(None, {'input': input_array})[0]
            
            if output.shape != (1, 22):
                print(f"   ❌ Wrong output shape: {output.shape}, expected (1, 22)")
                return False
            
            print(f"   ✓ Single image inference: shape {output.shape}")
            
            # Test batch (dynamic batch dimension)
            print("   ⏳ Testing dynamic batch dimension...")
            batch_input = np.random.randn(8, 3, 224, 224).astype(np.float32)
            batch_output = session.run(None, {'input': batch_input})[0]
            
            if batch_output.shape != (8, 22):
                print(f"   ❌ Wrong batch output shape: {batch_output.shape}, expected (8, 22)")
                return False
            
            print(f"   ✓ Batch inference: shape {batch_output.shape}")
            print("   ✅ ONNX Runtime tests passed!")
            
        except ImportError:
            print("   ⚠️  ONNX Runtime not installed (skip inference test)")
            print("      Install with: pip install onnxruntime")
        
        return True
    
    except Exception as e:
        print(f"   ❌ ONNX export failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_output_parity(
    pytorch_model: nn.Module,
    torchscript_path: Path,
    onnx_path: Path,
    num_samples: int = 5
) -> bool:
    """Test output parity between all formats."""
    print("\n🧪 Testing output parity...")
    
    all_passed = True
    
    # Load TorchScript
    try:
        traced_model = torch.jit.load(str(torchscript_path))
        traced_model.eval()
        
        print(f"\n   📊 PyTorch vs TorchScript...")
        max_diffs = []
        
        with torch.no_grad():
            for i in range(num_samples):
                test_input = torch.randn(1, 3, 224, 224)
                
                pytorch_output = pytorch_model(test_input)
                torchscript_output = traced_model(test_input)
                
                diff = torch.abs(pytorch_output - torchscript_output)
                max_diff = diff.max().item()
                max_diffs.append(max_diff)
        
        max_diff_overall = max(max_diffs)
        passed = max_diff_overall < 1e-5
        status = "✅ PASSED" if passed else "❌ FAILED"
        
        print(f"      {status} - Max difference: {max_diff_overall:.6e}")
        all_passed = all_passed and passed
    
    except Exception as e:
        print(f"      ❌ TorchScript parity test failed: {e}")
        all_passed = False
    
    # Test ONNX
    try:
        import onnxruntime as ort
        
        session = ort.InferenceSession(str(onnx_path))
        
        print(f"\n   📊 PyTorch vs ONNX...")
        max_diffs = []
        
        with torch.no_grad():
            for i in range(num_samples):
                test_input = torch.randn(1, 3, 224, 224)
                
                pytorch_output = pytorch_model(test_input).numpy()
                onnx_output = session.run(None, {'input': test_input.numpy()})[0]
                
                diff = np.abs(pytorch_output - onnx_output)
                max_diff = diff.max()
                max_diffs.append(max_diff)
        
        max_diff_overall = max(max_diffs)
        passed = max_diff_overall < 1e-5
        status = "✅ PASSED" if passed else "❌ FAILED"
        
        print(f"      {status} - Max difference: {max_diff_overall:.6e}")
        all_passed = all_passed and passed
    
    except ImportError:
        print(f"\n   ⚠️  ONNX Runtime not installed (skip parity test)")
    except Exception as e:
        print(f"      ❌ ONNX parity test failed: {e}")
        all_passed = False
    
    return all_passed


def main():
    """Run all export tests."""
    print("="*70)
    print("🧪 CropShield AI Export Test Suite")
    print("="*70 + "\n")
    
    # Create temporary directory
    temp_dir = Path(tempfile.mkdtemp())
    print(f"📁 Using temporary directory: {temp_dir}\n")
    
    try:
        # Create dummy model
        model = create_dummy_model()
        
        # Save checkpoint
        checkpoint_path = temp_dir / "test_model.pth"
        save_dummy_checkpoint(model, checkpoint_path)
        
        # Test TorchScript export
        torchscript_path = temp_dir / "test_model_scripted.pt"
        torchscript_passed = test_torchscript_export(model, torchscript_path)
        
        # Test ONNX export
        onnx_path = temp_dir / "test_model.onnx"
        onnx_passed = test_onnx_export(model, onnx_path)
        
        # Test output parity
        parity_passed = test_output_parity(
            model,
            torchscript_path,
            onnx_path
        )
        
        # Print summary
        print("\n" + "="*70)
        print("📊 TEST SUMMARY")
        print("="*70)
        
        tests = {
            'TorchScript Export': torchscript_passed,
            'ONNX Export': onnx_passed,
            'Output Parity': parity_passed
        }
        
        for test_name, passed in tests.items():
            status = "✅ PASSED" if passed else "❌ FAILED"
            print(f"   {test_name}: {status}")
        
        all_passed = all(tests.values())
        
        print("\n" + "="*70)
        if all_passed:
            print("✅ ALL TESTS PASSED!")
            print("\n💡 Ready to use export script:")
            print("   python scripts/export_model.py --model models/cropshield_cnn.pth")
        else:
            print("❌ SOME TESTS FAILED")
            print("\n⚠️  Please check the error messages above")
        print("="*70 + "\n")
        
        return 0 if all_passed else 1
    
    finally:
        # Cleanup
        print(f"🧹 Cleaning up temporary directory...")
        shutil.rmtree(temp_dir)
        print("   ✓ Done\n")


if __name__ == '__main__':
    sys.exit(main())
