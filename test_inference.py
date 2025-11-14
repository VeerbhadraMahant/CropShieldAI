"""
Test script for PyTorch inference functionality.

This script tests the predict.py module without requiring a trained model.
It creates a dummy model and verifies all inference functions work correctly.

Usage:
    python test_inference.py
"""

import torch
import torch.nn as nn
from PIL import Image
import numpy as np
from pathlib import Path
import json

# Import inference functions
from predict import (
    get_device,
    load_class_mapping,
    load_model,
    load_model_once,
    predict_disease,
    predict_batch,
    format_predictions
)

# Import model
from model_custom_cnn import CropShieldCNN


def create_dummy_model_and_data():
    """
    Create dummy model, image, and class mapping for testing.
    """
    print("="*60)
    print("CREATING TEST DATA")
    print("="*60)
    
    # Create test directory
    test_dir = Path('test_inference_data')
    test_dir.mkdir(exist_ok=True)
    
    # 1. Create dummy model
    num_classes = 5
    model = CropShieldCNN(num_classes=num_classes)
    
    # Save model
    model_path = test_dir / 'test_model.pth'
    torch.save({
        'model_state_dict': model.state_dict(),
        'model_type': 'custom',
        'num_classes': num_classes,
        'epoch': 10
    }, model_path)
    print(f"✅ Created dummy model: {model_path}")
    
    # 2. Create class mapping
    class_to_idx = {
        'Class_A': 0,
        'Class_B': 1,
        'Class_C': 2,
        'Class_D': 3,
        'Class_E': 4
    }
    
    class_mapping_path = test_dir / 'class_to_idx.json'
    with open(class_mapping_path, 'w') as f:
        json.dump(class_to_idx, f, indent=2)
    print(f"✅ Created class mapping: {class_mapping_path}")
    
    # 3. Create dummy image
    image = Image.new('RGB', (224, 224), color=(100, 150, 200))
    image_path = test_dir / 'test_image.jpg'
    image.save(image_path)
    print(f"✅ Created test image: {image_path}")
    
    print(f"{'='*60}\n")
    
    return model_path, image_path, test_dir


def test_device_detection():
    """Test device detection."""
    print("="*60)
    print("TEST 1: Device Detection")
    print("="*60)
    
    device = get_device()
    print(f"✅ Device detected: {device}")
    print(f"{'='*60}\n")
    
    return device


def test_class_mapping_loading(test_dir):
    """Test class mapping loading."""
    print("="*60)
    print("TEST 2: Class Mapping Loading")
    print("="*60)
    
    class_names = load_class_mapping(test_dir)
    print(f"✅ Loaded {len(class_names)} classes:")
    for i, name in enumerate(class_names):
        print(f"   {i}: {name}")
    print(f"{'='*60}\n")
    
    return class_names


def test_model_loading(model_path, device):
    """Test model loading."""
    print("="*60)
    print("TEST 3: Model Loading")
    print("="*60)
    
    model, class_names = load_model(str(model_path), device, verbose=True)
    print(f"✅ Model loaded successfully")
    print(f"✅ Model in eval mode: {not model.training}")
    print(f"{'='*60}\n")
    
    return model, class_names


def test_model_caching(model_path):
    """Test model caching."""
    print("="*60)
    print("TEST 4: Model Caching")
    print("="*60)
    
    # First call (should load)
    print("First call (loading)...")
    model1, class_names1, device1 = load_model_once(str(model_path))
    
    # Second call (should use cache)
    print("\nSecond call (cached)...")
    model2, class_names2, device2 = load_model_once(str(model_path))
    
    # Verify same objects
    assert model1 is model2, "❌ Models should be the same object"
    assert class_names1 is class_names2, "❌ Class names should be the same object"
    
    print(f"✅ Model caching works correctly")
    print(f"{'='*60}\n")
    
    return model1, class_names1, device1


def test_single_image_inference(image_path, model, class_names, device):
    """Test single image inference."""
    print("="*60)
    print("TEST 5: Single Image Inference")
    print("="*60)
    
    predictions = predict_disease(
        str(image_path),
        model,
        class_names,
        top_k=3,
        device=device
    )
    
    print(f"✅ Inference successful")
    print(f"   Top-3 predictions:")
    for i, (class_name, confidence) in enumerate(predictions, 1):
        print(f"   {i}. {class_name}: {confidence:.2%}")
    
    # Verify format
    assert len(predictions) == 3, "❌ Should return 3 predictions"
    assert all(isinstance(p[0], str) for p in predictions), "❌ Class names should be strings"
    assert all(isinstance(p[1], float) for p in predictions), "❌ Confidences should be floats"
    assert all(0 <= p[1] <= 1 for p in predictions), "❌ Confidences should be in [0, 1]"
    
    # Verify sorted by confidence
    confidences = [p[1] for p in predictions]
    assert confidences == sorted(confidences, reverse=True), "❌ Should be sorted by confidence"
    
    print(f"✅ Predictions format is correct")
    print(f"{'='*60}\n")
    
    return predictions


def test_batch_inference(image_path, model, class_names, device):
    """Test batch inference."""
    print("="*60)
    print("TEST 6: Batch Inference")
    print("="*60)
    
    # Create list of same image (for testing)
    image_paths = [str(image_path)] * 5
    
    batch_predictions = predict_batch(
        image_paths,
        model,
        class_names,
        top_k=2,
        batch_size=2,
        device=device
    )
    
    print(f"✅ Batch inference successful")
    print(f"   Processed {len(batch_predictions)} images")
    print(f"   Predictions per image:")
    for i, predictions in enumerate(batch_predictions, 1):
        print(f"   Image {i}: {predictions[0][0]} ({predictions[0][1]:.2%})")
    
    # Verify format
    assert len(batch_predictions) == 5, "❌ Should return 5 prediction lists"
    assert all(len(p) == 2 for p in batch_predictions), "❌ Each should have 2 predictions"
    
    print(f"✅ Batch inference format is correct")
    print(f"{'='*60}\n")


def test_format_predictions(predictions):
    """Test prediction formatting."""
    print("="*60)
    print("TEST 7: Prediction Formatting")
    print("="*60)
    
    # Format with all predictions
    formatted = format_predictions(predictions, show_all=True)
    print(formatted)
    
    # Format with top only
    formatted_top = format_predictions(predictions, show_all=False)
    print(formatted_top)
    
    print(f"✅ Formatting works correctly")
    print(f"{'='*60}\n")


def test_various_inputs(model, class_names, device, image_path):
    """Test various input types."""
    print("="*60)
    print("TEST 8: Various Input Types")
    print("="*60)
    
    # Test with PIL Image
    pil_image = Image.open(image_path)
    pred1 = predict_disease(pil_image, model, class_names, top_k=1, device=device)
    print(f"✅ PIL Image input: {pred1[0][0]} ({pred1[0][1]:.2%})")
    
    # Test with numpy array
    numpy_image = np.array(pil_image)
    pred2 = predict_disease(numpy_image, model, class_names, top_k=1, device=device)
    print(f"✅ Numpy array input: {pred2[0][0]} ({pred2[0][1]:.2%})")
    
    # Test with file path (string)
    pred3 = predict_disease(str(image_path), model, class_names, top_k=1, device=device)
    print(f"✅ File path input: {pred3[0][0]} ({pred3[0][1]:.2%})")
    
    # Predictions should be similar (same image)
    print(f"{'='*60}\n")


def cleanup(test_dir):
    """Clean up test data."""
    print("="*60)
    print("CLEANUP")
    print("="*60)
    
    import shutil
    if test_dir.exists():
        shutil.rmtree(test_dir)
        print(f"✅ Removed test directory: {test_dir}")
    
    print(f"{'='*60}\n")


def main():
    """
    Run all tests.
    """
    print("\n" + "="*60)
    print("PYTORCH INFERENCE TEST SUITE")
    print("="*60 + "\n")
    
    try:
        # Setup
        model_path, image_path, test_dir = create_dummy_model_and_data()
        
        # Run tests
        device = test_device_detection()
        class_names = test_class_mapping_loading(test_dir)
        model, class_names = test_model_loading(model_path, device)
        model, class_names, device = test_model_caching(model_path)
        predictions = test_single_image_inference(image_path, model, class_names, device)
        test_batch_inference(image_path, model, class_names, device)
        test_format_predictions(predictions)
        test_various_inputs(model, class_names, device, image_path)
        
        # Success
        print("="*60)
        print("✅ ALL TESTS PASSED!")
        print("="*60)
        print("\nThe predict.py module is working correctly.")
        print("You can now use it with your trained models.")
        print("\nNext steps:")
        print("1. Train your model (or use existing trained model)")
        print("2. Ensure models/class_to_idx.json exists")
        print("3. Run: python predict.py --image test.jpg --model models/cropshield_cnn.pth")
        print("="*60 + "\n")
        
    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        # Cleanup
        cleanup(test_dir)


if __name__ == '__main__':
    main()
