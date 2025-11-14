"""
Quick Example: PyTorch Inference Usage

This demonstrates the simplest way to use the predict.py module.
"""

from predict import load_model_once, predict_disease

# ============================================================================
# EXAMPLE 1: Single Image Prediction
# ============================================================================

print("="*60)
print("EXAMPLE 1: Single Image Prediction")
print("="*60)

# Load model once (cached for subsequent calls)
model, class_names, device = load_model_once('models/cropshield_cnn.pth')

# Predict disease
predictions = predict_disease(
    'test_image.jpg',           # Image path
    model,                       # Loaded model
    class_names,                 # Class names
    device,                      # Device (cuda/cpu)
    top_k=3                      # Top-3 predictions
)

# Display results
print("\nTop-3 Predictions:")
for class_name, confidence in predictions:
    disease = class_name.replace('__', ' - ').replace('_', ' ').title()
    print(f"  {disease}: {confidence:.2%}")

print()


# ============================================================================
# EXAMPLE 2: Batch Processing (Multiple Images)
# ============================================================================

print("="*60)
print("EXAMPLE 2: Batch Processing")
print("="*60)

from predict import predict_batch

# List of images
image_paths = [
    'image1.jpg',
    'image2.jpg',
    'image3.jpg'
]

# Batch inference (more efficient than loop)
batch_predictions = predict_batch(
    image_paths,
    model,
    class_names,
    device,
    top_k=1,
    batch_size=16
)

# Display results
print("\nBatch Results:")
for i, predictions in enumerate(batch_predictions):
    class_name, confidence = predictions[0]
    disease = class_name.replace('__', ' - ').replace('_', ' ').title()
    print(f"  Image {i+1}: {disease} ({confidence:.2%})")

print()


# ============================================================================
# EXAMPLE 3: Repeated Predictions (Model Cached)
# ============================================================================

print("="*60)
print("EXAMPLE 3: Cached Model (Instant Loading)")
print("="*60)

# This call is instant because model is cached
model, class_names, device = load_model_once('models/cropshield_cnn.pth')

# Run multiple predictions
for i in range(3):
    predictions = predict_disease(
        'test_image.jpg',
        model,
        class_names,
        device,
        top_k=1
    )
    class_name, confidence = predictions[0]
    print(f"  Prediction {i+1}: {class_name} ({confidence:.2%})")

print()


# ============================================================================
# EXAMPLE 4: Different Input Types
# ============================================================================

print("="*60)
print("EXAMPLE 4: Different Input Types")
print("="*60)

from PIL import Image
import numpy as np

# Load image once
image = Image.open('test_image.jpg')

# 1. PIL Image
pred1 = predict_disease(image, model, class_names, device, top_k=1)
print(f"  PIL Image: {pred1[0][0]} ({pred1[0][1]:.2%})")

# 2. Numpy array
numpy_array = np.array(image)
pred2 = predict_disease(numpy_array, model, class_names, device, top_k=1)
print(f"  Numpy Array: {pred2[0][0]} ({pred2[0][1]:.2%})")

# 3. File path
pred3 = predict_disease('test_image.jpg', model, class_names, device, top_k=1)
print(f"  File Path: {pred3[0][0]} ({pred3[0][1]:.2%})")

print("\n" + "="*60)
print("✅ All examples completed!")
print("="*60)
