"""
GradCAM Usage Examples
======================

Demonstrates how to use GradCAM for explainability visualization
with trained CropShield AI models.

Examples:
1. Basic GradCAM visualization
2. Top-k predictions comparison
3. Batch processing
4. Custom colormaps
5. Integration with inference pipeline
"""

import torch
from PIL import Image
import numpy as np
from pathlib import Path

# Import modules
from utils.gradcam import (
    GradCAM,
    generate_gradcam,
    generate_gradcam_visualization,
    get_target_layer,
    visualize_gradcam_grid,
    get_colormap_options,
    save_gradcam
)
from predict import load_model_once, predict_disease


# ============================================================================
# EXAMPLE 1: Basic GradCAM Visualization
# ============================================================================

def example_basic_gradcam():
    """
    Generate a simple GradCAM overlay for a single image.
    """
    print("="*60)
    print("EXAMPLE 1: Basic GradCAM Visualization")
    print("="*60)
    
    # Load model
    model, class_names, device = load_model_once('models/cropshield_cnn.pth')
    
    # Get target layer (last conv layer)
    target_layer = model.block4  # For Custom CNN
    # target_layer = get_target_layer(model, 'custom')  # Alternative
    
    # Generate GradCAM overlay
    overlay = generate_gradcam_visualization(
        model=model,
        image_input='test_leaf.jpg',
        target_layer=target_layer,
        device=device,
        alpha=0.5  # 50% overlay transparency
    )
    
    # Save result
    Image.fromarray(overlay).save('gradcam_basic.jpg')
    print(f"✅ Saved: gradcam_basic.jpg")
    print(f"{'='*60}\n")


# ============================================================================
# EXAMPLE 2: GradCAM with Prediction
# ============================================================================

def example_gradcam_with_prediction():
    """
    Generate GradCAM for the predicted class and show prediction confidence.
    """
    print("="*60)
    print("EXAMPLE 2: GradCAM with Prediction")
    print("="*60)
    
    # Load model
    model, class_names, device = load_model_once('models/cropshield_cnn.pth')
    
    # Run prediction
    image_path = 'test_leaf.jpg'
    predictions = predict_disease(image_path, model, class_names, device, top_k=3)
    
    # Display predictions
    print("\nPredictions:")
    for i, (class_name, confidence) in enumerate(predictions, 1):
        print(f"  {i}. {class_name}: {confidence:.2%}")
    
    # Get predicted class
    predicted_class = predictions[0][0]
    predicted_idx = class_names.index(predicted_class)
    
    # Generate GradCAM for predicted class
    overlay = generate_gradcam_visualization(
        model, image_path, model.block4, 
        class_idx=predicted_idx, 
        device=device
    )
    
    # Save with title
    import cv2
    overlay_with_title = overlay.copy()
    cv2.putText(
        overlay_with_title, 
        f"Predicted: {predicted_class} ({predictions[0][1]:.1%})",
        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2
    )
    
    Image.fromarray(overlay_with_title).save('gradcam_prediction.jpg')
    print(f"\n✅ Saved: gradcam_prediction.jpg")
    print(f"{'='*60}\n")


# ============================================================================
# EXAMPLE 3: Compare Top-K Predictions
# ============================================================================

def example_compare_topk():
    """
    Generate GradCAM for top-3 predictions to see what activates differently.
    """
    print("="*60)
    print("EXAMPLE 3: Compare Top-K Predictions")
    print("="*60)
    
    # Load model
    model, class_names, device = load_model_once('models/cropshield_cnn.pth')
    
    # Run prediction
    image_path = 'test_leaf.jpg'
    predictions = predict_disease(image_path, model, class_names, device, top_k=3)
    
    # Generate GradCAM for each top prediction
    overlays = []
    original = Image.open(image_path).resize((224, 224))
    overlays.append(np.array(original))
    
    for i, (class_name, confidence) in enumerate(predictions):
        class_idx = class_names.index(class_name)
        
        overlay = generate_gradcam_visualization(
            model, image_path, model.block4,
            class_idx=class_idx, device=device, alpha=0.5
        )
        
        # Add label
        import cv2
        cv2.putText(
            overlay, f"{class_name[:20]}", (10, 20),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1
        )
        cv2.putText(
            overlay, f"{confidence:.1%}", (10, 40),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1
        )
        
        overlays.append(overlay)
    
    # Create side-by-side comparison
    comparison = np.hstack(overlays)
    Image.fromarray(comparison).save('gradcam_topk_comparison.jpg')
    
    print(f"✅ Saved: gradcam_topk_comparison.jpg")
    print(f"   Columns: Original, Top-1, Top-2, Top-3")
    print(f"{'='*60}\n")


# ============================================================================
# EXAMPLE 4: Different Colormaps
# ============================================================================

def example_different_colormaps():
    """
    Show the same GradCAM with different color schemes.
    """
    print("="*60)
    print("EXAMPLE 4: Different Colormaps")
    print("="*60)
    
    # Load model
    model, class_names, device = load_model_once('models/cropshield_cnn.pth')
    
    # Get available colormaps
    colormaps = get_colormap_options()
    
    # Try different colormaps
    test_cmaps = ['jet', 'hot', 'viridis', 'rainbow', 'plasma']
    
    overlays = []
    for cmap_name in test_cmaps:
        import cv2
        cmap = colormaps[cmap_name]
        
        overlay = generate_gradcam_visualization(
            model, 'test_leaf.jpg', model.block4,
            device=device, colormap=cmap, alpha=0.5
        )
        
        # Add label
        cv2.putText(
            overlay, cmap_name, (10, 20),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2
        )
        
        overlays.append(overlay)
    
    # Create grid
    comparison = np.hstack(overlays)
    Image.fromarray(comparison).save('gradcam_colormaps.jpg')
    
    print(f"✅ Saved: gradcam_colormaps.jpg")
    print(f"   Colormaps: {', '.join(test_cmaps)}")
    print(f"{'='*60}\n")


# ============================================================================
# EXAMPLE 5: Batch Processing
# ============================================================================

def example_batch_processing():
    """
    Generate GradCAM for multiple images in a grid.
    """
    print("="*60)
    print("EXAMPLE 5: Batch Processing")
    print("="*60)
    
    # Load model
    model, class_names, device = load_model_once('models/cropshield_cnn.pth')
    
    # List of images
    image_paths = [
        'test_leaf1.jpg',
        'test_leaf2.jpg',
        'test_leaf3.jpg',
        'test_leaf4.jpg'
    ]
    
    # Generate grid
    grid = visualize_gradcam_grid(
        model, image_paths, model.block4, device=device,
        save_path='gradcam_batch_grid.jpg', alpha=0.5
    )
    
    print(f"✅ Saved: gradcam_batch_grid.jpg")
    print(f"   Images: {len(image_paths)}")
    print(f"{'='*60}\n")


# ============================================================================
# EXAMPLE 6: Advanced - Using GradCAM Class Directly
# ============================================================================

def example_advanced_usage():
    """
    Use GradCAM class directly for more control.
    """
    print("="*60)
    print("EXAMPLE 6: Advanced Usage")
    print("="*60)
    
    # Load model
    model, class_names, device = load_model_once('models/cropshield_cnn.pth')
    
    # Load and preprocess image
    from transforms import get_validation_transforms, IMAGENET_MEAN, IMAGENET_STD
    
    pil_image = Image.open('test_leaf.jpg')
    transform = get_validation_transforms(224, IMAGENET_MEAN, IMAGENET_STD)
    image_tensor = transform(pil_image).unsqueeze(0).to(device)
    
    # Create GradCAM instance
    gradcam = GradCAM(model, target_layer=model.block4, device=device)
    
    # Generate heatmap for specific class
    heatmap = gradcam(image_tensor, class_idx=5)
    
    print(f"✅ Heatmap generated")
    print(f"   Shape: {heatmap.shape}")
    print(f"   Range: [{heatmap.min():.3f}, {heatmap.max():.3f}]")
    
    # Create overlay manually
    original = np.array(pil_image.resize((224, 224)))
    overlay = gradcam.generate_heatmap_overlay(
        original, heatmap, alpha=0.6
    )
    
    Image.fromarray(overlay).save('gradcam_advanced.jpg')
    print(f"✅ Saved: gradcam_advanced.jpg")
    print(f"{'='*60}\n")


# ============================================================================
# EXAMPLE 7: Return Both Overlay and Heatmap
# ============================================================================

def example_return_heatmap():
    """
    Get both the overlay and raw heatmap for further processing.
    """
    print("="*60)
    print("EXAMPLE 7: Return Heatmap")
    print("="*60)
    
    # Load model
    model, class_names, device = load_model_once('models/cropshield_cnn.pth')
    
    # Generate with heatmap return
    overlay, heatmap = generate_gradcam_visualization(
        model, 'test_leaf.jpg', model.block4,
        device=device, return_heatmap=True
    )
    
    print(f"✅ Generated overlay and heatmap")
    print(f"   Overlay shape: {overlay.shape}")
    print(f"   Heatmap shape: {heatmap.shape}")
    
    # Save heatmap separately (as grayscale)
    heatmap_uint8 = (heatmap * 255).astype(np.uint8)
    Image.fromarray(heatmap_uint8, mode='L').save('gradcam_heatmap.jpg')
    
    # Save overlay
    Image.fromarray(overlay).save('gradcam_overlay.jpg')
    
    print(f"✅ Saved both overlay and heatmap")
    print(f"{'='*60}\n")


# ============================================================================
# EXAMPLE 8: Integration with Web API
# ============================================================================

def example_web_api_integration():
    """
    Example of integrating GradCAM with a Flask web API.
    """
    print("="*60)
    print("EXAMPLE 8: Web API Integration")
    print("="*60)
    
    example_code = '''
from flask import Flask, request, jsonify, send_file
from predict import load_model_once, predict_disease
from utils.gradcam import generate_gradcam_visualization
import io
from PIL import Image

app = Flask(__name__)

# Load model once at startup
model, class_names, device = load_model_once('models/cropshield_cnn.pth')

@app.route('/predict_with_gradcam', methods=['POST'])
def predict_with_gradcam():
    # Get image from request
    file = request.files['image']
    image = Image.open(io.BytesIO(file.read()))
    
    # Save temporarily
    temp_path = 'temp_upload.jpg'
    image.save(temp_path)
    
    # Run prediction
    predictions = predict_disease(
        temp_path, model, class_names, device, top_k=3
    )
    
    # Generate GradCAM
    overlay = generate_gradcam_visualization(
        model, temp_path, model.block4, device=device
    )
    
    # Save overlay
    overlay_path = 'temp_gradcam.jpg'
    Image.fromarray(overlay).save(overlay_path)
    
    # Return prediction and GradCAM
    return jsonify({
        'predictions': [
            {'disease': name, 'confidence': float(conf)}
            for name, conf in predictions
        ],
        'gradcam_url': '/gradcam/' + overlay_path
    })

@app.route('/gradcam/<path:filename>')
def serve_gradcam(filename):
    return send_file(filename, mimetype='image/jpeg')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
    '''
    
    print("Flask API example:")
    print(example_code)
    print(f"{'='*60}\n")


# ============================================================================
# Main
# ============================================================================

def main():
    """
    Run all examples (comment out as needed).
    """
    print("\n" + "="*60)
    print("GRADCAM USAGE EXAMPLES")
    print("="*60 + "\n")
    
    print("⚠️  Note: These examples require a trained model at:")
    print("    models/cropshield_cnn.pth")
    print("\n" + "="*60 + "\n")
    
    # Uncomment examples to run:
    # example_basic_gradcam()
    # example_gradcam_with_prediction()
    # example_compare_topk()
    # example_different_colormaps()
    # example_batch_processing()
    # example_advanced_usage()
    # example_return_heatmap()
    example_web_api_integration()
    
    print("✅ Examples complete!")
    print("\nFor more information, see:")
    print("  - utils/gradcam.py (main module)")
    print("  - test_gradcam.py (test suite)")
    print("="*60 + "\n")


if __name__ == '__main__':
    main()
