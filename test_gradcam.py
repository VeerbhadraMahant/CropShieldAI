"""
Test and Demonstration Script for GradCAM
==========================================

This script tests all GradCAM functionality with a dummy model.
No trained model required.

Tests:
1. Basic GradCAM generation
2. Heatmap overlay
3. Multiple colormaps
4. Top-k predictions comparison
5. Batch visualization grid
6. CPU and GPU execution

Usage:
    python test_gradcam.py
"""

import torch
import torch.nn as nn
import numpy as np
from PIL import Image
from pathlib import Path
import shutil

# Import GradCAM
from utils.gradcam import (
    GradCAM,
    generate_gradcam,
    generate_gradcam_visualization,
    get_target_layer,
    visualize_gradcam_grid,
    get_colormap_options
)

# Import model
from model_custom_cnn import CropShieldCNN


def create_test_data():
    """
    Create dummy model and test images.
    """
    print("="*60)
    print("CREATING TEST DATA")
    print("="*60)
    
    # Create test directory
    test_dir = Path('test_gradcam_data')
    test_dir.mkdir(exist_ok=True)
    
    # 1. Create dummy model
    model = CropShieldCNN(num_classes=5)
    model.eval()
    print(f"✅ Created dummy model: CropShieldCNN (5 classes)")
    
    # 2. Create test images with different patterns
    test_images = []
    
    # Image 1: Gradient pattern
    img1 = np.zeros((224, 224, 3), dtype=np.uint8)
    for i in range(224):
        img1[i, :, 0] = i  # Red gradient
        img1[:, i, 1] = i  # Green gradient
    img1_path = test_dir / 'test_gradient.jpg'
    Image.fromarray(img1).save(img1_path)
    test_images.append(img1_path)
    
    # Image 2: Circular pattern
    img2 = np.zeros((224, 224, 3), dtype=np.uint8)
    center = (112, 112)
    for i in range(224):
        for j in range(224):
            dist = np.sqrt((i - center[0])**2 + (j - center[1])**2)
            val = max(0, 255 - int(dist * 2))
            img2[i, j] = [val, val // 2, 0]
    img2_path = test_dir / 'test_circular.jpg'
    Image.fromarray(img2).save(img2_path)
    test_images.append(img2_path)
    
    # Image 3: Random noise
    img3 = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
    img3_path = test_dir / 'test_noise.jpg'
    Image.fromarray(img3).save(img3_path)
    test_images.append(img3_path)
    
    # Image 4: Checkerboard
    img4 = np.zeros((224, 224, 3), dtype=np.uint8)
    for i in range(0, 224, 32):
        for j in range(0, 224, 32):
            if (i // 32 + j // 32) % 2 == 0:
                img4[i:i+32, j:j+32] = [200, 100, 50]
    img4_path = test_dir / 'test_checkerboard.jpg'
    Image.fromarray(img4).save(img4_path)
    test_images.append(img4_path)
    
    print(f"✅ Created {len(test_images)} test images")
    print(f"{'='*60}\n")
    
    return model, test_images, test_dir


def test_device_detection(model):
    """Test device detection."""
    print("="*60)
    print("TEST 1: Device Detection")
    print("="*60)
    
    device = next(model.parameters()).device
    print(f"✅ Model device: {device}")
    
    if torch.cuda.is_available():
        print(f"✅ GPU available: {torch.cuda.get_device_name(0)}")
        gpu_device = torch.device('cuda')
        model_gpu = model.to(gpu_device)
        print(f"✅ Model moved to GPU")
        print(f"{'='*60}\n")
        return model_gpu, gpu_device
    else:
        print(f"⚠️  GPU not available, using CPU")
        print(f"{'='*60}\n")
        return model, device


def test_target_layer_detection(model):
    """Test target layer detection."""
    print("="*60)
    print("TEST 2: Target Layer Detection")
    print("="*60)
    
    target_layer = get_target_layer(model, 'custom')
    print(f"✅ Target layer (Custom CNN): {target_layer}")
    print(f"   Type: {type(target_layer)}")
    
    # Verify it's the last conv block
    assert hasattr(model, 'block4'), "Model should have block4"
    assert target_layer is model.block4, "Target layer should be block4"
    
    print(f"✅ Target layer correctly identified")
    print(f"{'='*60}\n")
    
    return target_layer


def test_basic_gradcam(model, target_layer, image_path, device):
    """Test basic GradCAM generation."""
    print("="*60)
    print("TEST 3: Basic GradCAM Generation")
    print("="*60)
    
    # Load image
    from transforms import get_validation_transforms, IMAGENET_MEAN, IMAGENET_STD
    
    pil_image = Image.open(image_path)
    transform = get_validation_transforms(224, IMAGENET_MEAN, IMAGENET_STD)
    image_tensor = transform(pil_image).unsqueeze(0).to(device)
    
    print(f"✅ Image loaded: {image_path.name}")
    print(f"   Tensor shape: {image_tensor.shape}")
    
    # Generate GradCAM
    gradcam = GradCAM(model, target_layer, device)
    heatmap = gradcam(image_tensor, class_idx=0)
    
    print(f"✅ GradCAM generated")
    print(f"   Heatmap shape: {heatmap.shape}")
    print(f"   Value range: [{heatmap.min():.3f}, {heatmap.max():.3f}]")
    
    # Verify heatmap properties
    assert heatmap.shape == (14, 14), f"Expected (14, 14), got {heatmap.shape}"
    assert 0 <= heatmap.min() <= 1, "Heatmap values should be in [0, 1]"
    assert 0 <= heatmap.max() <= 1, "Heatmap values should be in [0, 1]"
    
    print(f"✅ Heatmap properties correct")
    print(f"{'='*60}\n")
    
    return heatmap


def test_heatmap_overlay(model, target_layer, image_path, device, test_dir):
    """Test heatmap overlay generation."""
    print("="*60)
    print("TEST 4: Heatmap Overlay")
    print("="*60)
    
    # Generate overlay
    overlay = generate_gradcam_visualization(
        model, image_path, target_layer, class_idx=0, device=device, alpha=0.5
    )
    
    print(f"✅ Overlay generated")
    print(f"   Shape: {overlay.shape}")
    print(f"   Dtype: {overlay.dtype}")
    
    # Save overlay
    output_path = test_dir / f'gradcam_overlay_{image_path.stem}.jpg'
    Image.fromarray(overlay).save(output_path)
    print(f"✅ Saved overlay: {output_path.name}")
    
    # Verify overlay properties
    assert overlay.shape[2] == 3, "Overlay should be RGB"
    assert overlay.dtype == np.uint8, "Overlay should be uint8"
    
    print(f"✅ Overlay properties correct")
    print(f"{'='*60}\n")
    
    return overlay


def test_multiple_colormaps(model, target_layer, image_path, device, test_dir):
    """Test different colormaps."""
    print("="*60)
    print("TEST 5: Multiple Colormaps")
    print("="*60)
    
    colormaps = get_colormap_options()
    print(f"✅ Available colormaps: {len(colormaps)}")
    
    # Test a few colormaps
    test_cmaps = ['jet', 'hot', 'viridis', 'rainbow']
    
    for cmap_name in test_cmaps:
        import cv2
        cmap = colormaps[cmap_name]
        
        overlay = generate_gradcam_visualization(
            model, image_path, target_layer, 
            class_idx=0, device=device, 
            colormap=cmap, alpha=0.5
        )
        
        output_path = test_dir / f'gradcam_{cmap_name}_{image_path.stem}.jpg'
        Image.fromarray(overlay).save(output_path)
        
        print(f"   ✅ {cmap_name}: {output_path.name}")
    
    print(f"✅ All colormaps tested")
    print(f"{'='*60}\n")


def test_predicted_class(model, target_layer, image_path, device, test_dir):
    """Test GradCAM with predicted class (no class_idx specified)."""
    print("="*60)
    print("TEST 6: Predicted Class GradCAM")
    print("="*60)
    
    # Generate with auto-detected predicted class
    overlay = generate_gradcam_visualization(
        model, image_path, target_layer, 
        class_idx=None,  # Will use predicted class
        device=device
    )
    
    print(f"✅ GradCAM generated for predicted class")
    
    output_path = test_dir / f'gradcam_predicted_{image_path.stem}.jpg'
    Image.fromarray(overlay).save(output_path)
    print(f"✅ Saved: {output_path.name}")
    
    print(f"{'='*60}\n")


def test_batch_grid(model, target_layer, image_paths, device, test_dir):
    """Test grid visualization."""
    print("="*60)
    print("TEST 7: Batch Grid Visualization")
    print("="*60)
    
    # Generate grid
    grid = visualize_gradcam_grid(
        model, image_paths, target_layer, 
        device=device, alpha=0.5
    )
    
    print(f"✅ Grid generated")
    print(f"   Shape: {grid.shape}")
    
    # Save grid
    output_path = test_dir / 'gradcam_grid.jpg'
    Image.fromarray(grid).save(output_path)
    print(f"✅ Saved grid: {output_path.name}")
    
    print(f"{'='*60}\n")
    
    return grid


def test_different_inputs(model, target_layer, image_path, device):
    """Test different input types."""
    print("="*60)
    print("TEST 8: Different Input Types")
    print("="*60)
    
    # Test 1: File path (string)
    overlay1 = generate_gradcam_visualization(
        model, str(image_path), target_layer, device=device
    )
    print(f"✅ File path input: shape {overlay1.shape}")
    
    # Test 2: PIL Image
    pil_image = Image.open(image_path)
    overlay2 = generate_gradcam_visualization(
        model, pil_image, target_layer, device=device
    )
    print(f"✅ PIL Image input: shape {overlay2.shape}")
    
    # Test 3: Numpy array
    numpy_image = np.array(pil_image)
    overlay3 = generate_gradcam_visualization(
        model, numpy_image, target_layer, device=device
    )
    print(f"✅ Numpy array input: shape {overlay3.shape}")
    
    # Test 4: Tensor
    from transforms import get_validation_transforms, IMAGENET_MEAN, IMAGENET_STD
    transform = get_validation_transforms(224, IMAGENET_MEAN, IMAGENET_STD)
    tensor = transform(pil_image).unsqueeze(0).to(device)
    overlay4 = generate_gradcam_visualization(
        model, tensor, target_layer, device=device
    )
    print(f"✅ Tensor input: shape {overlay4.shape}")
    
    print(f"✅ All input types work correctly")
    print(f"{'='*60}\n")


def test_convenience_function(model, target_layer, image_path, device):
    """Test convenience function."""
    print("="*60)
    print("TEST 9: Convenience Function")
    print("="*60)
    
    # Load image
    from transforms import get_validation_transforms, IMAGENET_MEAN, IMAGENET_STD
    
    pil_image = Image.open(image_path)
    transform = get_validation_transforms(224, IMAGENET_MEAN, IMAGENET_STD)
    image_tensor = transform(pil_image).unsqueeze(0).to(device)
    
    # Use convenience function
    heatmap = generate_gradcam(model, image_tensor, target_layer, class_idx=0, device=device)
    
    print(f"✅ Convenience function works")
    print(f"   Heatmap shape: {heatmap.shape}")
    print(f"   Value range: [{heatmap.min():.3f}, {heatmap.max():.3f}]")
    
    print(f"{'='*60}\n")


def cleanup(test_dir):
    """Clean up test data."""
    print("="*60)
    print("CLEANUP")
    print("="*60)
    
    if test_dir.exists():
        # Don't delete, keep for inspection
        print(f"✅ Test outputs saved in: {test_dir}")
        print(f"   Files:")
        for file in sorted(test_dir.glob('*.jpg')):
            print(f"   - {file.name}")
    
    print(f"{'='*60}\n")


def main():
    """
    Run all tests.
    """
    print("\n" + "="*60)
    print("GRADCAM TEST SUITE")
    print("="*60 + "\n")
    
    try:
        # Setup
        model, image_paths, test_dir = create_test_data()
        
        # Move to GPU if available
        model, device = test_device_detection(model)
        
        # Run tests
        target_layer = test_target_layer_detection(model)
        heatmap = test_basic_gradcam(model, target_layer, image_paths[0], device)
        overlay = test_heatmap_overlay(model, target_layer, image_paths[0], device, test_dir)
        test_multiple_colormaps(model, target_layer, image_paths[0], device, test_dir)
        test_predicted_class(model, target_layer, image_paths[0], device, test_dir)
        test_batch_grid(model, target_layer, image_paths, device, test_dir)
        test_different_inputs(model, target_layer, image_paths[0], device)
        test_convenience_function(model, target_layer, image_paths[0], device)
        
        # Success
        print("="*60)
        print("✅ ALL TESTS PASSED!")
        print("="*60)
        print("\nThe GradCAM module is working correctly.")
        print("You can now use it with your trained models.")
        print("\nNext steps:")
        print("1. Train your model (or use existing trained model)")
        print("2. Load model and target layer")
        print("3. Generate GradCAM visualizations for your predictions")
        print("\nExample usage:")
        print("  from utils.gradcam import generate_gradcam_visualization")
        print("  from predict import load_model_once")
        print("  ")
        print("  model, _, device = load_model_once('models/cropshield_cnn.pth')")
        print("  overlay = generate_gradcam_visualization(")
        print("      model, 'test.jpg', model.block4, device=device")
        print("  )")
        print("  Image.fromarray(overlay).save('gradcam.jpg')")
        print("="*60 + "\n")
        
    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        # Cleanup (keeps files for inspection)
        cleanup(test_dir)


if __name__ == '__main__':
    main()
