"""
Simple ONNX Inference Example

Minimal code for running inference with exported ONNX model.
For production deployment on CPU servers or edge devices.
"""

import numpy as np
from PIL import Image
from typing import Tuple, List
import json


def load_onnx_model(model_path: str):
    """
    Load ONNX model for inference.
    
    Args:
        model_path: Path to ONNX model file
        
    Returns:
        ONNXRuntime inference session
    """
    try:
        import onnxruntime as ort
    except ImportError:
        raise ImportError("Please install onnxruntime: pip install onnxruntime")
    
    # Create inference session (CPU only)
    session = ort.InferenceSession(
        model_path,
        providers=['CPUExecutionProvider']
    )
    
    return session


def preprocess_image(image_path: str, size: int = 224) -> np.ndarray:
    """
    Preprocess image for model input.
    
    Args:
        image_path: Path to image file
        size: Target size (default: 224)
        
    Returns:
        Preprocessed numpy array (1, 3, 224, 224)
    """
    # Load image
    img = Image.open(image_path).convert('RGB')
    
    # Resize
    img = img.resize((size, size), Image.BILINEAR)
    
    # Convert to numpy array
    img_array = np.array(img, dtype=np.float32)
    
    # Normalize (ImageNet stats)
    mean = np.array([0.485, 0.456, 0.406]) * 255
    std = np.array([0.229, 0.224, 0.225]) * 255
    img_array = (img_array - mean) / std
    
    # Transpose to (C, H, W)
    img_array = img_array.transpose(2, 0, 1)
    
    # Add batch dimension
    img_array = np.expand_dims(img_array, axis=0)
    
    return img_array


def predict(session, image_array: np.ndarray) -> Tuple[int, np.ndarray]:
    """
    Run inference on preprocessed image.
    
    Args:
        session: ONNXRuntime session
        image_array: Preprocessed image (1, 3, 224, 224)
        
    Returns:
        (predicted_class_id, class_probabilities)
    """
    # Run inference
    outputs = session.run(None, {'input': image_array})
    logits = outputs[0]
    
    # Softmax to get probabilities
    exp_logits = np.exp(logits - np.max(logits, axis=1, keepdims=True))
    probabilities = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)
    
    # Get predicted class
    predicted_class = np.argmax(probabilities, axis=1)[0]
    
    return predicted_class, probabilities[0]


def predict_image(
    model_path: str,
    image_path: str,
    class_names: List[str] = None
) -> dict:
    """
    Complete prediction pipeline.
    
    Args:
        model_path: Path to ONNX model
        image_path: Path to input image
        class_names: List of class names (optional)
        
    Returns:
        Dictionary with prediction results
    """
    # Load model
    session = load_onnx_model(model_path)
    
    # Preprocess image
    image_array = preprocess_image(image_path)
    
    # Predict
    predicted_class, probabilities = predict(session, image_array)
    
    # Get top 5 predictions
    top5_indices = np.argsort(probabilities)[-5:][::-1]
    top5_probs = probabilities[top5_indices]
    
    results = {
        'predicted_class': int(predicted_class),
        'confidence': float(probabilities[predicted_class]),
        'top5_classes': [int(idx) for idx in top5_indices],
        'top5_confidences': [float(prob) for prob in top5_probs]
    }
    
    # Add class names if provided
    if class_names:
        results['predicted_label'] = class_names[predicted_class]
        results['top5_labels'] = [class_names[idx] for idx in top5_indices]
    
    return results


# ============================================================
# EXAMPLE USAGE
# ============================================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='ONNX model inference')
    parser.add_argument('--model', type=str, default='models/cropshield_quantized.onnx',
                       help='Path to ONNX model')
    parser.add_argument('--image', type=str, required=True,
                       help='Path to input image')
    parser.add_argument('--export_info', type=str, default='models/export_info.json',
                       help='Path to export info (for class names)')
    
    args = parser.parse_args()
    
    # Load class names if available
    class_names = None
    try:
        with open(args.export_info, 'r') as f:
            export_info = json.load(f)
            class_names = export_info.get('class_names', None)
    except:
        print("⚠️  Could not load class names from export_info.json")
    
    # Run prediction
    print(f"\n🔍 Running inference...")
    print(f"   Model: {args.model}")
    print(f"   Image: {args.image}\n")
    
    results = predict_image(args.model, args.image, class_names)
    
    # Print results
    print(f"{'='*60}")
    print(f"📊 PREDICTION RESULTS")
    print(f"{'='*60}")
    
    if 'predicted_label' in results:
        print(f"\n🎯 Predicted: {results['predicted_label']}")
    else:
        print(f"\n🎯 Predicted class: {results['predicted_class']}")
    
    print(f"   Confidence: {results['confidence']*100:.2f}%")
    
    print(f"\n📈 Top 5 predictions:")
    for i in range(5):
        if 'top5_labels' in results:
            label = results['top5_labels'][i]
            conf = results['top5_confidences'][i]
            print(f"   {i+1}. {label}: {conf*100:.2f}%")
        else:
            class_id = results['top5_classes'][i]
            conf = results['top5_confidences'][i]
            print(f"   {i+1}. Class {class_id}: {conf*100:.2f}%")
    
    print(f"\n{'='*60}\n")
