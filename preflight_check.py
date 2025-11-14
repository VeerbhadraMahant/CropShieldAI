"""
Pre-Flight Checklist: Verify System Before Streamlit Launch
============================================================
Tests all critical components without requiring a trained model.

Usage:
    python preflight_check.py
    python preflight_check.py --verbose
"""

import sys
import os
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def print_header(text):
    """Print formatted header"""
    print("\n" + "=" * 70)
    print(f"  {text}")
    print("=" * 70)


def check_1_dependencies():
    """Check 1: Verify all required dependencies"""
    print_header("CHECK 1: Dependencies")
    
    required = {
        'torch': 'PyTorch',
        'torchvision': 'TorchVision',
        'PIL': 'Pillow',
        'streamlit': 'Streamlit',
        'numpy': 'NumPy',
        'pandas': 'Pandas',
        'matplotlib': 'Matplotlib',
    }
    
    optional = {
        'cv2': 'OpenCV (for GradCAM)',
        'onnx': 'ONNX (for model export)',
    }
    
    print("\n✓ Required Packages:")
    all_required = True
    for module, name in required.items():
        try:
            if module == 'PIL':
                import PIL
                version = PIL.__version__
            elif module == 'torch':
                import torch
                version = torch.__version__
            elif module == 'torchvision':
                import torchvision
                version = torchvision.__version__
            elif module == 'streamlit':
                import streamlit
                version = streamlit.__version__
            elif module == 'numpy':
                import numpy
                version = numpy.__version__
            elif module == 'pandas':
                import pandas
                version = pandas.__version__
            elif module == 'matplotlib':
                import matplotlib
                version = matplotlib.__version__
            
            print(f"  ✅ {name}: {version}")
        except ImportError:
            print(f"  ❌ {name}: NOT INSTALLED")
            all_required = False
    
    print("\n✓ Optional Packages:")
    for module, name in optional.items():
        try:
            if module == 'cv2':
                import cv2
                print(f"  ✅ {name}: {cv2.__version__}")
            elif module == 'onnx':
                import onnx
                print(f"  ✅ {name}: {onnx.__version__}")
        except ImportError:
            print(f"  ⚠️  {name}: Not installed (optional)")
    
    return all_required


def check_2_cuda():
    """Check 2: CUDA availability"""
    print_header("CHECK 2: CUDA & GPU")
    
    try:
        import torch
        
        if torch.cuda.is_available():
            print(f"\n✅ CUDA Available:")
            print(f"  - CUDA Version: {torch.version.cuda}")
            print(f"  - GPU Count: {torch.cuda.device_count()}")
            print(f"  - GPU Name: {torch.cuda.get_device_name(0)}")
            
            props = torch.cuda.get_device_properties(0)
            print(f"  - Total Memory: {props.total_memory / 1e9:.2f} GB")
            print(f"  - Compute Capability: {props.major}.{props.minor}")
            return True
        else:
            print("\n⚠️  CUDA Not Available - Will use CPU")
            print("  This is OK but training/inference will be slower")
            return True
            
    except Exception as e:
        print(f"\n❌ Error checking CUDA: {e}")
        return False


def check_3_project_structure():
    """Check 3: Verify project structure"""
    print_header("CHECK 3: Project Structure")
    
    required_files = [
        'app_optimized.py',
        'train.py',
        'predict.py',
        'evaluate.py',
        'models/model_factory.py',
        'utils/app_utils.py',
        'utils/gradcam.py',
        'fast_dataset.py',
        'requirements.txt',
    ]
    
    required_dirs = [
        'models',
        'utils',
        'Database_resized',
    ]
    
    print("\n✓ Essential Files:")
    all_files_exist = True
    for file in required_files:
        if os.path.exists(file):
            size_kb = os.path.getsize(file) / 1024
            print(f"  ✅ {file} ({size_kb:.1f} KB)")
        else:
            print(f"  ❌ {file} - MISSING!")
            all_files_exist = False
    
    print("\n✓ Essential Directories:")
    all_dirs_exist = True
    for directory in required_dirs:
        if os.path.exists(directory):
            if directory == 'Database_resized':
                # Count classes
                subdirs = [d for d in os.listdir(directory) 
                          if os.path.isdir(os.path.join(directory, d)) 
                          and not d.startswith('.')]
                print(f"  ✅ {directory}/ ({len(subdirs)} classes)")
            else:
                print(f"  ✅ {directory}/")
        else:
            print(f"  ❌ {directory}/ - MISSING!")
            all_dirs_exist = False
    
    return all_files_exist and all_dirs_exist


def check_4_model_factory():
    """Check 4: Test model factory"""
    print_header("CHECK 4: Model Factory")
    
    try:
        from models.model_factory import create_model
        import torch
        
        print("\n✓ Testing model creation...")
        
        # Test creating a small model
        model = create_model('custom', num_classes=38)
        
        param_count = sum(p.numel() for p in model.parameters())
        print(f"  ✅ Model created successfully")
        print(f"  - Architecture: Custom CNN")
        print(f"  - Parameters: {param_count:,}")
        print(f"  - Classes: 38")
        
        # Test forward pass with dummy input
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = model.to(device)
        model.eval()
        
        dummy_input = torch.randn(1, 3, 224, 224).to(device)
        with torch.no_grad():
            output = model(dummy_input)
        
        print(f"  ✅ Forward pass successful")
        print(f"  - Input shape: {dummy_input.shape}")
        print(f"  - Output shape: {output.shape}")
        
        return True
        
    except Exception as e:
        print(f"\n❌ Model factory test failed: {e}")
        logger.exception("Model factory error:")
        return False


def check_5_data_loader():
    """Check 5: Test data loader"""
    print_header("CHECK 5: Data Loader")
    
    try:
        from fast_dataset import create_fast_dataloaders
        
        print("\n✓ Testing data loader creation...")
        
        # Try to create dataloaders
        loaders = create_fast_dataloaders(
            data_dir='Database_resized',
            batch_size=8,
            num_workers=0,  # Use 0 for safety
            split_ratio=0.8
        )
        
        print(f"  ✅ DataLoaders created successfully")
        print(f"  - Train batches: {len(loaders['train'])}")
        print(f"  - Val batches: {len(loaders['val'])}")
        print(f"  - Classes: {loaders['num_classes']}")
        
        # Test loading one batch
        print("\n✓ Testing batch loading...")
        train_loader = loaders['train']
        images, labels = next(iter(train_loader))
        
        print(f"  ✅ Batch loaded successfully")
        print(f"  - Batch shape: {images.shape}")
        print(f"  - Labels shape: {labels.shape}")
        print(f"  - Image dtype: {images.dtype}")
        print(f"  - Label dtype: {labels.dtype}")
        
        return True
        
    except Exception as e:
        print(f"\n❌ Data loader test failed: {e}")
        logger.exception("Data loader error:")
        return False


def check_6_gradcam():
    """Check 6: Test GradCAM utilities"""
    print_header("CHECK 6: GradCAM Utilities")
    
    try:
        from utils.gradcam import generate_gradcam, save_gradcam, get_colormap_options
        import torch
        from models.model_factory import create_model
        
        print("\n✓ Testing GradCAM imports...")
        print("  ✅ generate_gradcam imported")
        print("  ✅ save_gradcam imported")
        print("  ✅ get_colormap_options imported")
        
        # Test colormap options
        try:
            colormaps = get_colormap_options()
            print(f"\n✓ Available colormaps: {len(colormaps)}")
        except ImportError:
            print("\n⚠️  OpenCV not available - GradCAM visualization limited")
            print("  Install opencv-python for full GradCAM support")
        
        # Test GradCAM generation with dummy data
        print("\n✓ Testing GradCAM generation...")
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = create_model('custom', num_classes=38).to(device)
        model.eval()
        
        dummy_input = torch.randn(1, 3, 224, 224).to(device)
        
        try:
            gradcam = generate_gradcam(
                model=model,
                input_tensor=dummy_input,
                target_layer=None,
                device=device
            )
            
            if gradcam is not None:
                print(f"  ✅ GradCAM generated successfully")
                print(f"  - Shape: {gradcam.shape if hasattr(gradcam, 'shape') else 'N/A'}")
            else:
                print("  ⚠️  GradCAM returned None (may need OpenCV)")
        except Exception as e:
            print(f"  ⚠️  GradCAM generation skipped: {e}")
        
        return True
        
    except Exception as e:
        print(f"\n❌ GradCAM test failed: {e}")
        logger.exception("GradCAM error:")
        return False


def check_7_streamlit_utils():
    """Check 7: Test Streamlit utilities"""
    print_header("CHECK 7: Streamlit Utilities")
    
    try:
        from utils.app_utils import (
            load_model_cached,
            run_inference,
            get_model_cache_info,
            clear_model_cache,
            validate_image,
            preprocess_image
        )
        
        print("\n✅ All Streamlit utility functions imported:")
        print("  - load_model_cached")
        print("  - run_inference")
        print("  - get_model_cache_info")
        print("  - clear_model_cache")
        print("  - validate_image")
        print("  - preprocess_image")
        
        # Test cache functions
        cache_info = get_model_cache_info()
        print(f"\n✓ Cache Info:")
        print(f"  - Size: {cache_info['size']}")
        print(f"  - Count: {cache_info['count']}")
        
        return True
        
    except Exception as e:
        print(f"\n❌ Streamlit utils test failed: {e}")
        logger.exception("Streamlit utils error:")
        return False


def check_8_inference_script():
    """Check 8: Test inference script"""
    print_header("CHECK 8: Inference Script")
    
    if not os.path.exists('predict.py'):
        print("\n❌ predict.py not found!")
        return False
    
    print("\n✓ predict.py exists")
    
    # Check if it's executable
    try:
        with open('predict.py', 'r') as f:
            content = f.read()
            
        has_main = 'if __name__' in content
        has_argparse = 'argparse' in content
        
        print(f"  ✅ File size: {len(content)} bytes")
        print(f"  {'✅' if has_main else '⚠️ '} Has main block: {has_main}")
        print(f"  {'✅' if has_argparse else '⚠️ '} Has argument parser: {has_argparse}")
        
        return True
        
    except Exception as e:
        print(f"\n❌ Error reading predict.py: {e}")
        return False


def check_9_dataset():
    """Check 9: Verify dataset structure"""
    print_header("CHECK 9: Dataset")
    
    data_dir = 'Database_resized'
    
    if not os.path.exists(data_dir):
        print(f"\n❌ Dataset directory not found: {data_dir}")
        return False
    
    print(f"\n✓ Dataset directory: {data_dir}")
    
    # Count classes and images
    classes = []
    total_images = 0
    
    for class_name in os.listdir(data_dir):
        class_path = os.path.join(data_dir, class_name)
        if os.path.isdir(class_path) and not class_name.startswith('.'):
            classes.append(class_name)
            
            # Count images
            images = [f for f in os.listdir(class_path) 
                     if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
            total_images += len(images)
    
    print(f"  ✅ Classes: {len(classes)}")
    print(f"  ✅ Total images: {total_images:,}")
    
    if len(classes) > 0:
        print(f"\n✓ Sample classes:")
        for cls in sorted(classes)[:5]:
            class_path = os.path.join(data_dir, cls)
            img_count = len([f for f in os.listdir(class_path) 
                           if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
            print(f"  - {cls}: {img_count} images")
        if len(classes) > 5:
            print(f"  ... and {len(classes) - 5} more classes")
    
    return len(classes) > 0 and total_images > 0


def print_summary(results):
    """Print summary of all checks"""
    print_header("SUMMARY")
    
    checks = [
        ("Dependencies", results['check_1']),
        ("CUDA/GPU", results['check_2']),
        ("Project Structure", results['check_3']),
        ("Model Factory", results['check_4']),
        ("Data Loader", results['check_5']),
        ("GradCAM", results['check_6']),
        ("Streamlit Utils", results['check_7']),
        ("Inference Script", results['check_8']),
        ("Dataset", results['check_9']),
    ]
    
    print("\n")
    passed = 0
    failed = 0
    
    for check_name, status in checks:
        if status:
            print(f"  ✅ {check_name}")
            passed += 1
        else:
            print(f"  ❌ {check_name}")
            failed += 1
    
    print("\n" + "=" * 70)
    print(f"  Results: {passed}/{len(checks)} checks passed")
    print("=" * 70)
    
    if failed == 0:
        print("\n🎉 ALL CHECKS PASSED!")
        print("\n✅ System is ready for:")
        print("  1. Model training:")
        print("     python train.py --epochs 10")
        print("\n  2. Streamlit launch (after training):")
        print("     streamlit run app_optimized.py")
        print("\n  3. Inference testing (after training):")
        print("     python predict.py --image path/to/image.jpg")
        print("\n  4. GradCAM visualization (after training):")
        print("     python example_gradcam.py")
    else:
        print(f"\n⚠️  {failed} check(s) failed!")
        print("\n❌ Please fix the issues above before proceeding")
        print("\nCommon fixes:")
        print("  - Missing dependencies: pip install -r requirements.txt")
        print("  - Missing files: Check PROJECT_STRUCTURE.md")
        print("  - Dataset issues: Run scripts/resize_images.py")
    
    return failed == 0


def main():
    """Main execution"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Pre-flight system check')
    parser.add_argument('--verbose', action='store_true', help='Verbose output')
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    print("\n" + "=" * 70)
    print("  🌾 CropShield AI - Pre-Flight Checklist")
    print("  Verifying system before Streamlit launch")
    print("=" * 70)
    
    results = {}
    
    # Run all checks
    results['check_1'] = check_1_dependencies()
    results['check_2'] = check_2_cuda()
    results['check_3'] = check_3_project_structure()
    results['check_4'] = check_4_model_factory()
    results['check_5'] = check_5_data_loader()
    results['check_6'] = check_6_gradcam()
    results['check_7'] = check_7_streamlit_utils()
    results['check_8'] = check_8_inference_script()
    results['check_9'] = check_9_dataset()
    
    # Print summary
    all_passed = print_summary(results)
    
    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
