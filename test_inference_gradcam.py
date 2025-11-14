"""
Test Script: Model Inference + GradCAM Verification
===================================================
Tests both inference and GradCAM functionality locally before Streamlit launch.

Usage:
    python test_inference_gradcam.py
    python test_inference_gradcam.py --image path/to/test/image.jpg
    python test_inference_gradcam.py --model path/to/model.pth
"""

import sys
import os
import logging
from pathlib import Path
import torch
import torch.nn as nn
from PIL import Image
import json

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Check CV2 availability
try:
    import cv2
    CV2_AVAILABLE = True
    logger.info(f"✓ OpenCV available: {cv2.__version__}")
except ImportError:
    CV2_AVAILABLE = False
    logger.warning("✗ OpenCV not available - GradCAM visualization will be limited")


def print_section(title):
    """Print formatted section header"""
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70)


def check_dependencies():
    """Check all required dependencies"""
    print_section("Step 1: Dependency Check")
    
    dependencies = {
        'torch': torch.__version__,
        'PIL': Image.__version__,
    }
    
    if CV2_AVAILABLE:
        dependencies['opencv'] = cv2.__version__
    
    print("\n✓ Core Dependencies:")
    for name, version in dependencies.items():
        print(f"  - {name}: {version}")
    
    # Check CUDA
    if torch.cuda.is_available():
        print(f"\n✓ CUDA Available:")
        print(f"  - Version: {torch.version.cuda}")
        print(f"  - GPU: {torch.cuda.get_device_name(0)}")
        print(f"  - Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    else:
        print("\n⚠ CUDA Not Available - Using CPU")
    
    return True


def find_test_image():
    """Find a test image in the dataset"""
    print_section("Step 2: Finding Test Image")
    
    # Look for test images in common locations
    search_paths = [
        "Database_resized/test",
        "Database_resized/train",
        "Database/test",
        "Database/train",
        "test_images",
    ]
    
    for search_path in search_paths:
        if not os.path.exists(search_path):
            continue
        
        # Walk through directories to find first image
        for root, dirs, files in os.walk(search_path):
            for file in files:
                if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                    image_path = os.path.join(root, file)
                    class_name = os.path.basename(root)
                    
                    print(f"\n✓ Found test image:")
                    print(f"  - Path: {image_path}")
                    print(f"  - Class: {class_name}")
                    
                    # Check image properties
                    img = Image.open(image_path)
                    print(f"  - Size: {img.size}")
                    print(f"  - Mode: {img.mode}")
                    
                    return image_path, class_name
    
    print("\n✗ No test images found in common locations")
    print("  Please provide an image path using --image flag")
    return None, None


def load_model(model_path=None):
    """Load the trained model"""
    print_section("Step 3: Loading Model")
    
    # Find model if not specified
    if model_path is None:
        model_candidates = [
            "best_model.pth",
            "models/best_model.pth",
            "checkpoints/best_model.pth",
        ]
        
        for candidate in model_candidates:
            if os.path.exists(candidate):
                model_path = candidate
                break
    
    if model_path is None or not os.path.exists(model_path):
        print("\n✗ Model not found!")
        print("  Searched locations:")
        for candidate in model_candidates:
            print(f"    - {candidate}")
        print("\n  Please train a model first or specify path with --model flag")
        return None, None
    
    print(f"\n✓ Loading model from: {model_path}")
    
    try:
        # Load checkpoint
        checkpoint = torch.load(model_path, map_location='cpu')
        
        # Extract model info
        if isinstance(checkpoint, dict):
            num_classes = checkpoint.get('num_classes', 38)
            model_arch = checkpoint.get('model_name', 'custom')
            print(f"  - Architecture: {model_arch}")
            print(f"  - Classes: {num_classes}")
            
            if 'epoch' in checkpoint:
                print(f"  - Epoch: {checkpoint['epoch']}")
            if 'val_acc' in checkpoint:
                print(f"  - Validation Accuracy: {checkpoint['val_acc']:.2f}%")
        else:
            num_classes = 38
            model_arch = 'custom'
        
        # Create model
        from models.model_factory import create_model
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = create_model(model_arch, num_classes=num_classes)
        
        # Load weights
        if isinstance(checkpoint, dict):
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
        
        model = model.to(device)
        model.eval()
        
        print(f"  - Device: {device}")
        print(f"  - Parameters: {sum(p.numel() for p in model.parameters()):,}")
        print("\n✓ Model loaded successfully!")
        
        return model, device
        
    except Exception as e:
        print(f"\n✗ Failed to load model: {e}")
        logger.exception("Model loading error:")
        return None, None


def load_class_mapping():
    """Load class index to name mapping"""
    mapping_files = [
        "class_mapping.json",
        "models/class_mapping.json",
    ]
    
    for mapping_file in mapping_files:
        if os.path.exists(mapping_file):
            with open(mapping_file, 'r') as f:
                return json.load(f)
    
    # Return default mapping if file not found
    logger.warning("Class mapping file not found, using indices")
    return {str(i): f"Class_{i}" for i in range(38)}


def test_inference(model, device, image_path):
    """Test model inference"""
    print_section("Step 4: Testing Inference")
    
    try:
        # Load and preprocess image
        from torchvision import transforms
        
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225])
        ])
        
        print(f"\n✓ Loading image: {image_path}")
        image = Image.open(image_path).convert('RGB')
        print(f"  - Original size: {image.size}")
        
        # Transform
        input_tensor = transform(image).unsqueeze(0).to(device)
        print(f"  - Tensor shape: {input_tensor.shape}")
        print(f"  - Tensor device: {input_tensor.device}")
        
        # Inference
        print("\n✓ Running inference...")
        with torch.no_grad():
            outputs = model(input_tensor)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
            confidence, predicted_idx = torch.max(probabilities, 1)
        
        predicted_class = predicted_idx.item()
        confidence_score = confidence.item() * 100
        
        print(f"\n✓ Inference Results:")
        print(f"  - Predicted Class Index: {predicted_class}")
        print(f"  - Confidence: {confidence_score:.2f}%")
        
        # Load class names
        class_mapping = load_class_mapping()
        class_name = class_mapping.get(str(predicted_class), f"Class_{predicted_class}")
        print(f"  - Predicted Class Name: {class_name}")
        
        # Show top-5 predictions
        print(f"\n✓ Top-5 Predictions:")
        top5_prob, top5_idx = torch.topk(probabilities, 5, dim=1)
        for i, (idx, prob) in enumerate(zip(top5_idx[0], top5_prob[0]), 1):
            idx_val = idx.item()
            prob_val = prob.item() * 100
            cls_name = class_mapping.get(str(idx_val), f"Class_{idx_val}")
            print(f"  {i}. {cls_name} ({prob_val:.2f}%)")
        
        print("\n✅ INFERENCE TEST PASSED!")
        return True, input_tensor, image
        
    except Exception as e:
        print(f"\n✗ Inference failed: {e}")
        logger.exception("Inference error:")
        return False, None, None


def test_gradcam(model, device, input_tensor, original_image):
    """Test GradCAM functionality"""
    print_section("Step 5: Testing GradCAM")
    
    if not CV2_AVAILABLE:
        print("\n⚠ Skipping GradCAM test - OpenCV not available")
        print("  Install opencv-python to enable GradCAM visualization")
        return False
    
    try:
        from utils.gradcam import generate_gradcam
        
        print("\n✓ Generating GradCAM...")
        
        # Generate GradCAM
        gradcam_output = generate_gradcam(
            model=model,
            input_tensor=input_tensor,
            target_layer=None,  # Auto-detect
            device=device
        )
        
        if gradcam_output is None:
            print("✗ GradCAM generation returned None")
            return False
        
        print(f"  - GradCAM shape: {gradcam_output.shape if hasattr(gradcam_output, 'shape') else 'unknown'}")
        print(f"  - Type: {type(gradcam_output)}")
        
        # Try to save GradCAM
        output_dir = "test_outputs"
        os.makedirs(output_dir, exist_ok=True)
        
        from utils.gradcam import save_gradcam
        
        output_path = os.path.join(output_dir, "test_gradcam.jpg")
        save_gradcam(
            gradcam=gradcam_output,
            original_image=original_image,
            save_path=output_path,
            alpha=0.4
        )
        
        if os.path.exists(output_path):
            print(f"\n✓ GradCAM saved to: {output_path}")
            file_size = os.path.getsize(output_path) / 1024
            print(f"  - File size: {file_size:.2f} KB")
            print("\n✅ GRADCAM TEST PASSED!")
            return True
        else:
            print("\n✗ GradCAM file was not saved")
            return False
        
    except ImportError as e:
        print(f"\n✗ Import error: {e}")
        print("  Make sure utils/gradcam.py is available")
        return False
    except Exception as e:
        print(f"\n✗ GradCAM generation failed: {e}")
        logger.exception("GradCAM error:")
        return False


def test_streamlit_utils():
    """Test Streamlit utility functions"""
    print_section("Step 6: Testing Streamlit Utils")
    
    try:
        from utils.app_utils import load_model_cached, run_inference
        
        print("\n✓ Successfully imported Streamlit utilities:")
        print("  - load_model_cached")
        print("  - run_inference")
        
        print("\n✓ Checking cache functions...")
        from utils.app_utils import get_model_cache_info, clear_model_cache
        
        cache_info = get_model_cache_info()
        print(f"  - Cache size: {cache_info['size']}")
        print(f"  - Cached models: {cache_info['count']}")
        
        print("\n✅ STREAMLIT UTILS TEST PASSED!")
        return True
        
    except ImportError as e:
        print(f"\n✗ Import error: {e}")
        print("  Make sure utils/app_utils.py is available")
        return False
    except Exception as e:
        print(f"\n✗ Streamlit utils test failed: {e}")
        logger.exception("Streamlit utils error:")
        return False


def print_summary(results):
    """Print final summary"""
    print_section("FINAL SUMMARY")
    
    checks = [
        ("Dependencies", results.get('dependencies', False)),
        ("Test Image Found", results.get('test_image', False)),
        ("Model Loading", results.get('model_loading', False)),
        ("Inference", results.get('inference', False)),
        ("GradCAM", results.get('gradcam', False)),
        ("Streamlit Utils", results.get('streamlit_utils', False)),
    ]
    
    print("\n")
    all_passed = True
    for check_name, passed in checks:
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"  {status}  {check_name}")
        if not passed:
            all_passed = False
    
    print("\n" + "=" * 70)
    if all_passed:
        print("  🎉 ALL TESTS PASSED - READY FOR STREAMLIT LAUNCH!")
    else:
        print("  ⚠️  SOME TESTS FAILED - REVIEW ERRORS ABOVE")
    print("=" * 70)
    
    if all_passed:
        print("\n🚀 Next Steps:")
        print("  1. Launch Streamlit app:")
        print("     streamlit run app_optimized.py")
        print("\n  2. Or run full deployment validation:")
        print("     python validate_deployment.py --verbose")
        print("\n  3. Test inference API:")
        print("     python predict.py --image path/to/image.jpg")
    
    return all_passed


def main():
    """Main test execution"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Test Inference + GradCAM')
    parser.add_argument('--image', type=str, help='Path to test image')
    parser.add_argument('--model', type=str, help='Path to model checkpoint')
    args = parser.parse_args()
    
    print("\n" + "=" * 70)
    print("  🌾 CropShield AI - Inference + GradCAM Test Suite")
    print("=" * 70)
    
    results = {}
    
    # Step 1: Check dependencies
    results['dependencies'] = check_dependencies()
    if not results['dependencies']:
        print("\n❌ Dependency check failed!")
        return 1
    
    # Step 2: Find test image
    image_path = args.image
    if image_path is None:
        image_path, _ = find_test_image()
    
    results['test_image'] = image_path is not None
    if not results['test_image']:
        print("\n❌ No test image available!")
        return 1
    
    # Step 3: Load model
    model, device = load_model(args.model)
    results['model_loading'] = model is not None
    if not results['model_loading']:
        print("\n❌ Model loading failed!")
        return 1
    
    # Step 4: Test inference
    success, input_tensor, original_image = test_inference(model, device, image_path)
    results['inference'] = success
    if not success:
        print("\n❌ Inference test failed!")
        return 1
    
    # Step 5: Test GradCAM
    results['gradcam'] = test_gradcam(model, device, input_tensor, original_image)
    if not results['gradcam'] and CV2_AVAILABLE:
        print("\n⚠️ GradCAM test failed (but not critical)")
    
    # Step 6: Test Streamlit utils
    results['streamlit_utils'] = test_streamlit_utils()
    
    # Print summary
    all_passed = print_summary(results)
    
    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
