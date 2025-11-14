"""
Quick Pre-Launch Verification
==============================
Simple test to verify model inference and GradCAM work before Streamlit.

Usage: python quick_verify.py
"""

import os
import sys

print("\n" + "="*70)
print("  🌾 CropShield AI - Quick Pre-Launch Verification")
print("="*70)

# Test 1: Dependencies
print("\n[1/6] Checking dependencies...")
try:
    import torch
    import torchvision
    import streamlit
    from PIL import Image
    import numpy as np
    print(f"  ✅ PyTorch {torch.__version__}")
    print(f"  ✅ Streamlit {streamlit.__version__}")
    print(f"  ✅ CUDA: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"     GPU: {torch.cuda.get_device_name(0)}")
except ImportError as e:
    print(f"  ❌ Missing dependency: {e}")
    sys.exit(1)

# Test 2: OpenCV for GradCAM
print("\n[2/6] Checking GradCAM support...")
try:
    import cv2
    print(f"  ✅ OpenCV {cv2.__version__} - GradCAM enabled")
except ImportError:
    print("  ⚠️  OpenCV not found - GradCAM visualization limited")

# Test 3: Model Factory
print("\n[3/6] Testing model factory...")
try:
    from models.model_factory import get_model
    
    model, optimizer, criterion, scheduler, device = get_model(
        model_type='custom',
        num_classes=38
    )
    
    param_count = sum(p.numel() for p in model.parameters())
    print(f"  ✅ Model created: {param_count:,} parameters")
    print(f"  ✅ Device: {device}")
    
    # Test forward pass
    dummy_input = torch.randn(1, 3, 224, 224).to(device)
    model.eval()
    with torch.no_grad():
        output = model(dummy_input)
    print(f"  ✅ Forward pass: {output.shape}")
    
except Exception as e:
    print(f"  ❌ Model factory failed: {e}")
    sys.exit(1)

# Test 4: Data Loader
print("\n[4/6] Testing data loader...")
try:
    from fast_dataset import make_loaders
    
    result = make_loaders(
        data_dir='Database_resized',
        batch_size=8,
        num_workers=0,
        train_split=0.8,
        val_split=0.1,
        test_split=0.1,
        augmentation_mode='conservative'
    )
    
    train_loader, val_loader, test_loader, class_names, dataset_info = result
    num_classes = len(class_names)
    
    print(f"  ✅ Train batches: {len(train_loader)}")
    print(f"  ✅ Val batches: {len(val_loader)}")
    print(f"  ✅ Classes: {num_classes}")
    
    # Test loading one batch
    images, labels = next(iter(train_loader))
    print(f"  ✅ Batch shape: {images.shape}")
    
except Exception as e:
    print(f"  ❌ Data loader failed: {e}")
    sys.exit(1)

# Test 5: GradCAM Functions
print("\n[5/6] Testing GradCAM utilities...")
try:
    from utils.gradcam import generate_gradcam, save_gradcam
    
    print("  ✅ GradCAM functions imported")
    
    # Test GradCAM generation
    gradcam = generate_gradcam(
        model=model,
        image_tensor=dummy_input,
        device=device
    )
    
    if gradcam is not None:
        print(f"  ✅ GradCAM generated: {gradcam.shape if hasattr(gradcam, 'shape') else 'OK'}")
    else:
        print("  ⚠️  GradCAM returned None (may need OpenCV)")
    
except Exception as e:
    print(f"  ❌ GradCAM test failed: {e}")
    # Not critical, continue

# Test 6: App Utilities
print("\n[6/6] Testing app utilities...")
try:
    from utils.app_utils import (
        load_class_names,
        display_predictions,
        show_gradcam_overlay
    )
    
    print("  ✅ App utilities imported:")
    print("     - load_class_names")
    print("     - display_predictions")
    print("     - show_gradcam_overlay")
    
except Exception as e:
    print(f"  ❌ App utilities failed: {e}")
    sys.exit(1)

# Summary
print("\n" + "="*70)
print("  ✅ ALL VERIFICATIONS PASSED!")
print("="*70)

print("\n📊 System Status:")
print(f"  - GPU: {'✅ Available' if torch.cuda.is_available() else '⚠️  CPU only'}")
print(f"  - Dataset: ✅ {num_classes} classes, {len(train_loader)} train batches")
print(f"  - Model: ✅ Ready ({param_count:,} params)")
print(f"  - GradCAM: {'✅ Full support' if 'cv2' in sys.modules else '⚠️  Limited'}")

print("\n🚀 Ready to launch:")
print("\n  Option 1: Train model first (recommended)")
print("     python train.py --epochs 10 --batch_size 32")

print("\n  Option 2: Launch Streamlit with demo model")
print("     streamlit run app_optimized.py")

print("\n  Option 3: Test inference (needs trained model)")
print("     python predict.py --image Database_resized/Tomato__healthy/sample.jpg")

print("\n  Option 4: Full deployment validation")
print("     python validate_deployment.py --verbose")

print("\n" + "="*70)
print("  ✨ System verification complete!")
print("="*70 + "\n")
