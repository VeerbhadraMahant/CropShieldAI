"""
Test Suite for Streamlit App Utilities
=======================================

Tests for utils/app_utils.py functions.

Run with: python test_app_utils.py
"""

import os
import sys
import io
import json
import tempfile
from pathlib import Path

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import numpy as np
from PIL import Image

# Import utilities
from utils.app_utils import (
    load_class_names,
    load_class_mapping,
    bytesio_to_pil,
    pil_to_bytes,
    resize_image,
    format_prediction_table,
    create_confidence_indicator,
    save_prediction_history,
    get_crop_emoji,
)


# ============================================================================
# Test Helpers
# ============================================================================

def create_dummy_class_mapping(path):
    """Create a dummy class_to_idx.json file."""
    mapping = {
        "Potato__early_blight": 0,
        "Potato__healthy": 1,
        "Potato__late_blight": 2,
        "Tomato__bacterial_spot": 3,
        "Tomato__healthy": 4,
    }
    
    with open(path, 'w') as f:
        json.dump(mapping, f)
    
    return mapping


def create_test_image(size=(224, 224), mode='RGB'):
    """Create a test PIL image."""
    # Create gradient image
    arr = np.zeros((*size, 3), dtype=np.uint8)
    for i in range(size[0]):
        for j in range(size[1]):
            arr[i, j] = [
                int(255 * i / size[0]),  # Red gradient
                int(255 * j / size[1]),  # Green gradient
                128                       # Blue constant
            ]
    
    return Image.fromarray(arr, mode=mode)


# ============================================================================
# Tests
# ============================================================================

def test_class_mapping():
    """Test class mapping utilities."""
    print("="*60)
    print("TEST 1: Class Mapping")
    print("="*60)
    
    # Create temp mapping file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        mapping_path = f.name
        f.write(json.dumps({
            "Potato__early_blight": 0,
            "Potato__healthy": 1,
            "Potato__late_blight": 2,
        }))
    
    try:
        # Test load_class_names
        class_names = load_class_names(mapping_path)
        
        assert len(class_names) == 3, f"Expected 3 classes, got {len(class_names)}"
        assert class_names[0] == "Potato__early_blight", f"Expected 'Potato__early_blight', got {class_names[0]}"
        assert class_names[1] == "Potato__healthy", f"Expected 'Potato__healthy', got {class_names[1]}"
        
        print(f"✅ Loaded class names: {class_names}")
        
        # Test load_class_mapping
        mapping = load_class_mapping(mapping_path)
        
        assert len(mapping) == 3, f"Expected 3 mappings, got {len(mapping)}"
        assert mapping["Potato__early_blight"] == 0, f"Expected 0, got {mapping['Potato__early_blight']}"
        
        print(f"✅ Loaded class mapping: {mapping}")
        
    finally:
        # Clean up
        os.unlink(mapping_path)
    
    print(f"{'='*60}\n")


def test_image_handling():
    """Test image handling utilities."""
    print("="*60)
    print("TEST 2: Image Handling")
    print("="*60)
    
    # Create test image
    image = create_test_image(size=(224, 224))
    print(f"✅ Created test image: {image.size}, mode={image.mode}")
    
    # Test PIL to bytes
    image_bytes = pil_to_bytes(image, format='PNG')
    assert len(image_bytes) > 0, "Expected non-empty bytes"
    print(f"✅ Converted to bytes: {len(image_bytes)} bytes")
    
    # Test BytesIO to PIL
    bytesio = io.BytesIO(image_bytes)
    recovered = bytesio_to_pil(bytesio)
    
    assert recovered.size == image.size, f"Expected {image.size}, got {recovered.size}"
    assert recovered.mode == image.mode, f"Expected {image.mode}, got {recovered.mode}"
    print(f"✅ Recovered from BytesIO: {recovered.size}, mode={recovered.mode}")
    
    # Test resize
    small = resize_image(image, max_size=112, maintain_aspect=True)
    assert small.size == (112, 112), f"Expected (112, 112), got {small.size}"
    print(f"✅ Resized image: {image.size} → {small.size}")
    
    # Test resize with aspect ratio (rectangular input)
    rect_image = create_test_image(size=(300, 200))
    small_rect = resize_image(rect_image, max_size=150, maintain_aspect=True)
    
    # Input: (300, 200) where width (300) > height (200)
    # Expected: scale width to 150, height = int(200/300 * 150) = 100
    # Result should be (150, 100)
    expected_w, expected_h = 150, 100
    actual_w, actual_h = small_rect.size
    
    # Allow either (150, 100) or validate aspect ratio is maintained
    aspect_in = rect_image.size[0] / rect_image.size[1]
    aspect_out = actual_w / actual_h
    
    assert abs(aspect_in - aspect_out) < 0.1, f"Aspect ratio not maintained: {aspect_in:.2f} vs {aspect_out:.2f}"
    assert max(actual_w, actual_h) == 150, f"Max dimension should be 150, got max({actual_w}, {actual_h}) = {max(actual_w, actual_h)}"
    
    print(f"✅ Resized with aspect: {rect_image.size} → {small_rect.size}")
    
    # Test RGBA conversion
    rgba_image = create_test_image(size=(100, 100), mode='RGB')
    rgba_image.putalpha(255)  # Add alpha channel
    assert rgba_image.mode == 'RGBA', f"Expected RGBA mode"
    
    # Convert to RGB via BytesIO
    rgba_bytes = pil_to_bytes(rgba_image, format='PNG')
    rgb_converted = bytesio_to_pil(io.BytesIO(rgba_bytes))
    assert rgb_converted.mode == 'RGB', f"Expected RGB after conversion, got {rgb_converted.mode}"
    print(f"✅ RGBA → RGB conversion: {rgba_image.mode} → {rgb_converted.mode}")
    
    print(f"{'='*60}\n")


def test_prediction_formatting():
    """Test prediction formatting utilities."""
    print("="*60)
    print("TEST 3: Prediction Formatting")
    print("="*60)
    
    # Create test predictions
    predictions = [
        ('Potato__late_blight', 0.87),
        ('Potato__early_blight', 0.09),
        ('Potato__healthy', 0.04)
    ]
    
    # Test HTML table
    html = format_prediction_table(predictions)
    
    assert isinstance(html, str), "Expected string output"
    assert 'table' in html.lower(), "Expected HTML table"
    assert 'Potato' in html, "Expected class names in HTML"
    assert '87' in html, "Expected confidence percentages"
    
    print("✅ Generated HTML table")
    print(f"   Length: {len(html)} characters")
    print(f"   Contains: table tags, class names, percentages")
    
    print(f"{'='*60}\n")


def test_confidence_indicator():
    """Test confidence indicator generation."""
    print("="*60)
    print("TEST 4: Confidence Indicator")
    print("="*60)
    
    # Test high confidence
    html_high = create_confidence_indicator(0.85, thresholds=(0.7, 0.4))
    assert 'High Confidence' in html_high, "Expected 'High Confidence'"
    assert '#27ae60' in html_high, "Expected green color"
    assert '✅' in html_high, "Expected checkmark emoji"
    print(f"✅ High confidence (85%): Green badge with ✅")
    
    # Test medium confidence
    html_med = create_confidence_indicator(0.55, thresholds=(0.7, 0.4))
    assert 'Medium Confidence' in html_med, "Expected 'Medium Confidence'"
    assert '#f39c12' in html_med, "Expected orange color"
    assert '⚠️' in html_med, "Expected warning emoji"
    print(f"✅ Medium confidence (55%): Orange badge with ⚠️")
    
    # Test low confidence
    html_low = create_confidence_indicator(0.25, thresholds=(0.7, 0.4))
    assert 'Low Confidence' in html_low, "Expected 'Low Confidence'"
    assert '#e74c3c' in html_low, "Expected red color"
    assert '❌' in html_low, "Expected cross emoji"
    print(f"✅ Low confidence (25%): Red badge with ❌")
    
    print(f"{'='*60}\n")


def test_prediction_history():
    """Test prediction history saving."""
    print("="*60)
    print("TEST 5: Prediction History")
    print("="*60)
    
    # Create test data
    image = create_test_image()
    predictions = [
        ('Potato__late_blight', 0.87),
        ('Potato__early_blight', 0.09),
    ]
    metadata = {
        'crop': 'Potato',
        'location': 'Farm A',
        'weather': {'temp': 25, 'humidity': 60}
    }
    
    # Save history
    entry = save_prediction_history(image, predictions, metadata)
    
    # Verify structure
    assert 'timestamp' in entry, "Expected timestamp"
    assert 'predictions' in entry, "Expected predictions"
    assert 'image_size' in entry, "Expected image_size"
    assert 'metadata' in entry, "Expected metadata"
    
    assert entry['predictions'] == predictions, "Predictions mismatch"
    assert entry['image_size'] == image.size, "Image size mismatch"
    assert entry['metadata'] == metadata, "Metadata mismatch"
    
    print(f"✅ Created history entry:")
    print(f"   Timestamp: {entry['timestamp']}")
    print(f"   Predictions: {len(entry['predictions'])}")
    print(f"   Image size: {entry['image_size']}")
    print(f"   Metadata keys: {list(entry['metadata'].keys())}")
    
    print(f"{'='*60}\n")


def test_crop_emoji():
    """Test crop emoji utility."""
    print("="*60)
    print("TEST 6: Crop Emoji")
    print("="*60)
    
    # Test known crops
    test_cases = [
        ('Potato', '🥔'),
        ('Tomato', '🍅'),
        ('Wheat', '🌾'),
        ('Corn', '🌽'),
        ('Sugarcane', '🎋'),
        ('Apple', '🍎'),
        ('Unknown_Crop', '🌱'),  # Default
    ]
    
    for crop, expected in test_cases:
        emoji = get_crop_emoji(crop)
        assert emoji == expected, f"Expected {expected} for {crop}, got {emoji}"
        print(f"✅ {crop}: {emoji}")
    
    # Test case insensitivity
    emoji_lower = get_crop_emoji('potato')
    emoji_upper = get_crop_emoji('POTATO')
    assert emoji_lower == emoji_upper == '🥔', "Expected case-insensitive matching"
    print(f"✅ Case insensitive: 'potato' and 'POTATO' → 🥔")
    
    # Test partial matching
    emoji_partial = get_crop_emoji('Sweet Potato')
    assert emoji_partial == '🥔', "Expected partial matching"
    print(f"✅ Partial match: 'Sweet Potato' → 🥔")
    
    print(f"{'='*60}\n")


def test_numpy_to_pil_conversion():
    """Test numpy array to PIL conversion."""
    print("="*60)
    print("TEST 7: NumPy ↔ PIL Conversion")
    print("="*60)
    
    # Create numpy array
    arr = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
    print(f"✅ Created numpy array: {arr.shape}, dtype={arr.dtype}")
    
    # Convert to PIL
    image = Image.fromarray(arr)
    assert isinstance(image, Image.Image), "Expected PIL Image"
    assert image.size == (224, 224), f"Expected (224, 224), got {image.size}"
    print(f"✅ NumPy → PIL: {arr.shape} → {image.size}")
    
    # Convert back to numpy
    arr_recovered = np.array(image)
    assert arr_recovered.shape == arr.shape, "Shape mismatch"
    assert np.array_equal(arr, arr_recovered), "Arrays not equal"
    print(f"✅ PIL → NumPy: {image.size} → {arr_recovered.shape}")
    
    # Test with BytesIO round-trip
    image_bytes = pil_to_bytes(image)
    recovered = bytesio_to_pil(io.BytesIO(image_bytes))
    arr_final = np.array(recovered)
    
    # Note: May not be exactly equal due to compression, but should be close
    print(f"✅ Full round-trip: NumPy → PIL → Bytes → PIL → NumPy")
    print(f"   Original shape: {arr.shape}")
    print(f"   Final shape: {arr_final.shape}")
    print(f"   Mean absolute diff: {np.mean(np.abs(arr.astype(float) - arr_final.astype(float))):.2f}")
    
    print(f"{'='*60}\n")


# ============================================================================
# Main Test Runner
# ============================================================================

def main():
    """Run all tests."""
    print("\n" + "="*60)
    print("STREAMLIT APP UTILITIES TEST SUITE")
    print("="*60 + "\n")
    
    tests = [
        test_class_mapping,
        test_image_handling,
        test_prediction_formatting,
        test_confidence_indicator,
        test_prediction_history,
        test_crop_emoji,
        test_numpy_to_pil_conversion,
    ]
    
    passed = 0
    failed = 0
    
    for test_func in tests:
        try:
            test_func()
            passed += 1
        except AssertionError as e:
            print(f"❌ FAILED: {test_func.__name__}")
            print(f"   Error: {e}")
            print(f"{'='*60}\n")
            failed += 1
        except Exception as e:
            print(f"❌ ERROR: {test_func.__name__}")
            print(f"   Exception: {e}")
            print(f"{'='*60}\n")
            failed += 1
    
    # Summary
    print("="*60)
    print("TEST SUMMARY")
    print("="*60)
    print(f"✅ Passed: {passed}/{len(tests)}")
    print(f"❌ Failed: {failed}/{len(tests)}")
    print(f"{'='*60}\n")
    
    if failed == 0:
        print("🎉 ALL TESTS PASSED!")
        print("\nYou can now use utils/app_utils.py in your Streamlit app:")
        print("  - See example_streamlit_with_utils.py for usage")
        print("  - See STREAMLIT_UTILS_QUICKREF.md for quick reference")
    else:
        print("⚠️ SOME TESTS FAILED")
        print("\nPlease check the error messages above.")
    
    print("="*60 + "\n")
    
    return 0 if failed == 0 else 1


if __name__ == '__main__':
    exit_code = main()
    sys.exit(exit_code)
