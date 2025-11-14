# GradCAM Explainability Guide

**Complete guide to using Gradient-weighted Class Activation Mapping (GradCAM) for visual explainability in CropShield AI.**

---

## 📋 Table of Contents

1. [What is GradCAM?](#what-is-gradcam)
2. [Quick Start](#quick-start)
3. [Installation](#installation)
4. [Basic Usage](#basic-usage)
5. [Advanced Features](#advanced-features)
6. [Interpretation Guide](#interpretation-guide)
7. [Best Practices](#best-practices)
8. [Colormap Selection](#colormap-selection)
9. [Integration Examples](#integration-examples)
10. [Troubleshooting](#troubleshooting)
11. [API Reference](#api-reference)

---

## 🎯 What is GradCAM?

**Gradient-weighted Class Activation Mapping (GradCAM)** is a visualization technique that shows which regions of an image are most important for a CNN's prediction.

### How It Works

1. **Forward Pass**: Image → CNN → Prediction
2. **Backward Pass**: Compute gradients of target class w.r.t. last conv layer
3. **Importance Weights**: Global average pooling of gradients
4. **Weighted Activation**: Multiply activations by weights, sum, and apply ReLU
5. **Heatmap**: Normalize to [0, 1] and resize to original image size
6. **Overlay**: Blend heatmap with original image

### Mathematical Formulation

```
α_k = (1/Z) Σ_i Σ_j (∂y^c / ∂A^k_ij)    # Importance weights
L^c_GradCAM = ReLU(Σ_k α_k A^k)          # Weighted sum with ReLU
```

Where:
- `y^c`: Score for class c
- `A^k`: Activations of feature map k
- `α_k`: Importance weight for feature map k

### Why Use GradCAM?

✅ **Model Validation**: Verify model focuses on disease symptoms, not background  
✅ **Error Analysis**: Understand why model misclassified an image  
✅ **Trust Building**: Show users why model made a prediction  
✅ **Dataset Quality**: Identify annotation errors or spurious correlations  
✅ **Model Comparison**: Compare attention patterns between architectures

---

## 🚀 Quick Start

### 5-Minute Example

```python
from utils.gradcam import generate_gradcam_visualization
from predict import load_model_once

# Load model
model, class_names, device = load_model_once('models/cropshield_cnn.pth')

# Generate GradCAM
overlay = generate_gradcam_visualization(
    model, 
    'diseased_leaf.jpg', 
    model.block4,  # Last conv layer for Custom CNN
    device=device
)

# Save
from PIL import Image
Image.fromarray(overlay).save('gradcam_result.jpg')
```

**That's it!** You now have a visual explanation of the prediction.

---

## 📦 Installation

### Prerequisites

```bash
pip install torch torchvision opencv-python pillow numpy
```

### Files Required

- `utils/gradcam.py` - Main GradCAM module
- `predict.py` - Inference utilities
- `transforms.py` - Image preprocessing
- `models/cropshield_cnn.pth` - Trained model
- `models/class_to_idx.json` - Class mapping

---

## 📖 Basic Usage

### 1. Single Image GradCAM

```python
from utils.gradcam import generate_gradcam_visualization
from predict import load_model_once

# Load model
model, class_names, device = load_model_once('models/cropshield_cnn.pth')

# Generate GradCAM (auto-detects predicted class)
overlay = generate_gradcam_visualization(
    model, 'test_leaf.jpg', model.block4, device=device
)

# Save result
from PIL import Image
Image.fromarray(overlay).save('gradcam_output.jpg')
```

**Output**: RGB image (224x224x3) with heatmap overlay

### 2. GradCAM for Specific Class

```python
# Generate GradCAM for class index 5 (e.g., "Potato__late_blight")
overlay = generate_gradcam_visualization(
    model, 'test_leaf.jpg', model.block4, 
    class_idx=5, device=device
)
```

**Use Case**: Understand why model ranked a specific class highly

### 3. Get Both Overlay and Heatmap

```python
overlay, heatmap = generate_gradcam_visualization(
    model, 'test_leaf.jpg', model.block4, 
    device=device, return_heatmap=True
)

print(f"Overlay: {overlay.shape}")   # (224, 224, 3)
print(f"Heatmap: {heatmap.shape}")   # (224, 224)
print(f"Range: [{heatmap.min():.2f}, {heatmap.max():.2f}]")  # [0.00, 1.00]
```

**Use Case**: Further processing of raw heatmap

---

## 🔥 Advanced Features

### 1. Compare Top-K Predictions

See which regions activate for different disease predictions:

```python
from predict import predict_disease

# Get predictions
predictions = predict_disease('ambiguous_leaf.jpg', model, class_names, device, top_k=3)

# Generate GradCAM for each
overlays = []
for class_name, confidence in predictions:
    class_idx = class_names.index(class_name)
    
    overlay = generate_gradcam_visualization(
        model, 'ambiguous_leaf.jpg', model.block4,
        class_idx=class_idx, device=device
    )
    
    overlays.append((class_name, confidence, overlay))

# Save side-by-side comparison
import numpy as np
comparison = np.hstack([o[2] for o in overlays])
Image.fromarray(comparison).save('topk_comparison.jpg')
```

**Output**: Shows different activation patterns for each predicted class

### 2. Batch Grid Visualization

Process multiple images at once:

```python
from utils.gradcam import visualize_gradcam_grid

image_paths = [
    'potato_early_blight_1.jpg',
    'potato_early_blight_2.jpg',
    'potato_late_blight_1.jpg',
    'potato_late_blight_2.jpg'
]

grid = visualize_gradcam_grid(
    model, image_paths, model.block4, device=device,
    save_path='disease_analysis_grid.jpg'
)
```

**Output**: Grid layout showing GradCAM for all images

### 3. Different Colormaps

```python
import cv2
from utils.gradcam import get_colormap_options

colormaps = get_colormap_options()

for name, cmap in colormaps.items():
    overlay = generate_gradcam_visualization(
        model, 'test_leaf.jpg', model.block4,
        device=device, colormap=cmap, alpha=0.5
    )
    
    Image.fromarray(overlay).save(f'gradcam_{name}.jpg')
```

**Available Colormaps**: jet, hot, viridis, plasma, inferno, rainbow, cool, spring, summer, autumn, winter, bone

### 4. Custom Alpha (Overlay Transparency)

```python
# Subtle overlay (30% heatmap, 70% original)
overlay = generate_gradcam_visualization(
    model, 'test_leaf.jpg', model.block4,
    device=device, alpha=0.3
)

# Strong overlay (70% heatmap, 30% original)
overlay = generate_gradcam_visualization(
    model, 'test_leaf.jpg', model.block4,
    device=device, alpha=0.7
)
```

**Recommendation**: Use `alpha=0.4` to `alpha=0.6` for balanced visualization

### 5. Low-Level GradCAM API

For maximum control:

```python
from utils.gradcam import GradCAM
from transforms import get_validation_transforms, IMAGENET_MEAN, IMAGENET_STD

# Load and preprocess image
pil_image = Image.open('test_leaf.jpg')
transform = get_validation_transforms(224, IMAGENET_MEAN, IMAGENET_STD)
image_tensor = transform(pil_image).unsqueeze(0).to(device)

# Create GradCAM instance
gradcam = GradCAM(model, target_layer=model.block4, device=device)

# Generate heatmap
heatmap = gradcam(image_tensor, class_idx=None)  # None = predicted class

print(f"Heatmap shape: {heatmap.shape}")  # (14, 14)
print(f"Range: [{heatmap.min():.3f}, {heatmap.max():.3f}]")  # [0.000, 1.000]

# Create overlay
original = np.array(pil_image.resize((224, 224)))
overlay = gradcam.generate_heatmap_overlay(original, heatmap, alpha=0.5)

Image.fromarray(overlay).save('custom_gradcam.jpg')
```

---

## 🔍 Interpretation Guide

### What Do the Colors Mean?

| Color | Meaning | Interpretation |
|-------|---------|----------------|
| 🔴 **Red/Yellow** | High activation | Model focused on these regions |
| 🟡 **Orange** | Medium activation | Some attention |
| 🔵 **Blue/Green** | Low activation | Model ignored these regions |

### For Disease Classification

#### ✅ Good GradCAM (Model working correctly)

```
[Leaf image with brown lesions]
      ↓
[GradCAM shows RED over lesions, BLUE over healthy tissue]
      ↓
✅ Model correctly focuses on disease symptoms
```

**What to look for**:
- Red regions align with disease symptoms (lesions, spots, discoloration)
- Blue regions on healthy leaf tissue
- Activation concentrated, not scattered

#### ❌ Bad GradCAM (Model issues)

```
[Leaf image with lesions]
      ↓
[GradCAM shows RED on background, BLUE on lesions]
      ↓
⚠️ Model focuses on wrong regions (dataset bias)
```

**Common Issues**:
1. **Background Activation**: Model learns spurious patterns (soil color, lighting)
2. **Scattered Activation**: No clear focus (model uncertain)
3. **Edge Activation**: Model focuses on image borders (preprocessing artifacts)
4. **Uniform Activation**: Heatmap looks flat (layer too early, low resolution)

### Example Interpretations

#### Example 1: Late Blight Detection
```python
# Image: Potato leaf with dark brown lesions
overlay = generate_gradcam_visualization(model, 'potato_late_blight.jpg', ...)
```

**Expected GradCAM**:
- 🔴 Strong activation (red) on dark lesions
- 🔴 Activation on lesion edges (spreading pattern)
- 🔵 Low activation (blue) on healthy green tissue

**Interpretation**: Model correctly identifies lesion patterns characteristic of late blight

#### Example 2: Early Blight Detection
```python
# Image: Tomato leaf with concentric ring spots
overlay = generate_gradcam_visualization(model, 'tomato_early_blight.jpg', ...)
```

**Expected GradCAM**:
- 🔴 Strong activation on ring-shaped spots
- 🟡 Medium activation on yellowing surrounding tissue
- 🔵 Low activation on green leaf areas

**Interpretation**: Model recognizes distinctive "target spot" pattern of early blight

#### Example 3: Healthy Leaf
```python
# Image: Healthy green leaf
overlay = generate_gradcam_visualization(model, 'healthy_leaf.jpg', ...)
```

**Expected GradCAM**:
- 🟡 Distributed activation across leaf surface
- 🔵 No strong focal points
- Even, smooth heatmap

**Interpretation**: Model finds no disease symptoms; uniform attention indicates healthy classification

---

## 💡 Best Practices

### 1. Model Architecture Considerations

#### Custom CNN
```python
target_layer = model.block4  # Last conv layer (14x14 resolution)
```

#### EfficientNet-B0
```python
from utils.gradcam import get_target_layer
target_layer = get_target_layer(model, 'efficientnet_b0')
```

#### ResNet-18
```python
target_layer = get_target_layer(model, 'resnet18')
```

**Tip**: Use the last convolutional layer for best spatial resolution

### 2. Image Quality

✅ **Do**:
- Use high-resolution images (at least 224x224)
- Ensure good lighting and focus
- Center the leaf in the frame

❌ **Don't**:
- Use blurry or low-quality images
- Include excessive background
- Use heavily augmented validation images

### 3. Batch Processing

```python
# ❌ Inefficient: Loop over single images
for img in image_list:
    overlay = generate_gradcam_visualization(model, img, ...)

# ✅ Efficient: Use grid visualization
grid = visualize_gradcam_grid(model, image_list, target_layer, device=device)
```

### 4. Performance Tips

```python
# Use GPU for faster processing
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Cache model (don't reload repeatedly)
model, class_names, device = load_model_once('models/cropshield_cnn.pth')

# Process in batches if generating many GradCAMs
```

**Performance Benchmarks**:
- GPU (RTX 4060): ~10-20ms per image
- CPU (Intel i7): ~50-100ms per image

### 5. Validation Workflow

1. **Generate GradCAM** for random test samples
2. **Verify activations** align with disease symptoms
3. **Check misclassifications** - does GradCAM reveal why?
4. **Compare classes** - do similar diseases show different patterns?
5. **Document findings** for model improvements

---

## 🎨 Colormap Selection

### Choosing the Right Colormap

| Colormap | Best For | Characteristics |
|----------|----------|-----------------|
| **jet** | General use | High contrast, intuitive |
| **hot** | Medical/thermal | Black→Red→White progression |
| **viridis** | Publications | Perceptually uniform, colorblind-friendly |
| **plasma** | Presentations | Vibrant, perceptually uniform |
| **rainbow** | High detail | Many color gradations |

### Example Usage

```python
import cv2

# Medical/agricultural (recommended)
overlay = generate_gradcam_visualization(
    model, image, target_layer, device=device,
    colormap=cv2.COLORMAP_JET  # Default
)

# For publications (colorblind-friendly)
overlay = generate_gradcam_visualization(
    model, image, target_layer, device=device,
    colormap=cv2.COLORMAP_VIRIDIS
)

# For presentations (vibrant)
overlay = generate_gradcam_visualization(
    model, image, target_layer, device=device,
    colormap=cv2.COLORMAP_PLASMA
)
```

### Visualization Examples

```python
from utils.gradcam import get_colormap_options

# Get all available colormaps
colormaps = get_colormap_options()

# Generate comparison
for name, cmap in colormaps.items():
    overlay = generate_gradcam_visualization(
        model, 'test_leaf.jpg', model.block4,
        device=device, colormap=cmap
    )
    Image.fromarray(overlay).save(f'colormap_{name}.jpg')
```

---

## 🔗 Integration Examples

### 1. Flask Web API

```python
from flask import Flask, request, jsonify, send_file
from predict import load_model_once, predict_disease
from utils.gradcam import generate_gradcam_visualization
import io
from PIL import Image

app = Flask(__name__)

# Load model once at startup
model, class_names, device = load_model_once('models/cropshield_cnn.pth')

@app.route('/predict', methods=['POST'])
def predict_with_gradcam():
    # Get image
    file = request.files['image']
    image = Image.open(io.BytesIO(file.read()))
    image.save('temp.jpg')
    
    # Predict
    predictions = predict_disease('temp.jpg', model, class_names, device, top_k=3)
    
    # Generate GradCAM
    overlay = generate_gradcam_visualization(
        model, 'temp.jpg', model.block4, device=device
    )
    
    # Save overlay
    Image.fromarray(overlay).save('temp_gradcam.jpg')
    
    return jsonify({
        'predictions': [
            {'disease': name, 'confidence': float(conf)}
            for name, conf in predictions
        ],
        'gradcam_url': '/gradcam/temp_gradcam.jpg'
    })

@app.route('/gradcam/<path:filename>')
def serve_gradcam(filename):
    return send_file(filename, mimetype='image/jpeg')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
```

### 2. Streamlit Dashboard

```python
import streamlit as st
from predict import load_model_once, predict_disease
from utils.gradcam import generate_gradcam_visualization

# Title
st.title("CropShield AI - Explainable Disease Detection")

# Load model (cached)
@st.cache_resource
def get_model():
    return load_model_once('models/cropshield_cnn.pth')

model, class_names, device = get_model()

# Upload image
uploaded_file = st.file_uploader("Upload leaf image", type=['jpg', 'png'])

if uploaded_file:
    # Display original
    image = Image.open(uploaded_file)
    st.image(image, caption="Original Image", width=300)
    
    # Save temporarily
    image.save('temp.jpg')
    
    # Predict
    with st.spinner("Analyzing..."):
        predictions = predict_disease('temp.jpg', model, class_names, device, top_k=3)
    
    # Display predictions
    st.subheader("Predictions")
    for i, (disease, conf) in enumerate(predictions, 1):
        st.write(f"{i}. **{disease}**: {conf:.2%}")
    
    # Generate GradCAM
    with st.spinner("Generating explainability heatmap..."):
        overlay = generate_gradcam_visualization(
            model, 'temp.jpg', model.block4, device=device
        )
    
    # Display GradCAM
    st.subheader("Model Focus (GradCAM)")
    st.image(overlay, caption="Regions that influenced prediction", width=300)
    st.info("🔴 Red areas: Strong activation (model focused here)")
```

### 3. Batch Processing Script

```python
import os
from pathlib import Path
from tqdm import tqdm

# Load model
model, class_names, device = load_model_once('models/cropshield_cnn.pth')

# Process all images in directory
input_dir = Path('test_images')
output_dir = Path('gradcam_results')
output_dir.mkdir(exist_ok=True)

for img_path in tqdm(list(input_dir.glob('*.jpg'))):
    # Generate GradCAM
    overlay = generate_gradcam_visualization(
        model, str(img_path), model.block4, device=device
    )
    
    # Save
    output_path = output_dir / f"gradcam_{img_path.name}"
    Image.fromarray(overlay).save(output_path)

print(f"✅ Processed {len(list(input_dir.glob('*.jpg')))} images")
```

---

## 🐛 Troubleshooting

### Issue 1: Heatmap looks uniform (no clear activation)

**Symptoms**: GradCAM overlay is mostly one color

**Causes**:
- Wrong target layer (too early in network)
- Model not trained properly
- Image preprocessing mismatch

**Solutions**:
```python
# ✅ Use last conv layer
target_layer = model.block4  # Not model.block1!

# ✅ Verify model is loaded correctly
print(model)

# ✅ Check preprocessing matches training
from transforms import get_validation_transforms
transform = get_validation_transforms(224, IMAGENET_MEAN, IMAGENET_STD)
```

### Issue 2: Heatmap activates on background

**Symptoms**: Red regions on soil/background, blue on leaf

**Causes**:
- Dataset bias (spurious correlations)
- Model learns background patterns

**Solutions**:
```python
# Analyze multiple images to confirm
for img in test_images:
    overlay = generate_gradcam_visualization(model, img, ...)
    
# If consistent, retrain model with:
# - Better data augmentation (random backgrounds)
# - Cropped/masked images (remove background)
# - More diverse dataset
```

### Issue 3: CUDA out of memory

**Symptoms**: `RuntimeError: CUDA out of memory`

**Solutions**:
```python
# ✅ Use CPU for GradCAM generation
device = torch.device('cpu')

# ✅ Clear cache before processing
import torch
torch.cuda.empty_cache()

# ✅ Process images one at a time (not in batches)
```

### Issue 4: GradCAM shape mismatch

**Symptoms**: `ValueError: shapes not aligned`

**Causes**:
- Image size doesn't match model input
- Wrong target layer

**Solutions**:
```python
# ✅ Ensure correct image size
from transforms import get_validation_transforms
transform = get_validation_transforms(224, IMAGENET_MEAN, IMAGENET_STD)
image_tensor = transform(pil_image)

# ✅ Verify target layer has spatial dimensions
print(f"Target layer: {target_layer}")
# Should be Conv2d layer, not Linear!
```

### Issue 5: Different results each run

**Symptoms**: GradCAM heatmap changes on same image

**Causes**:
- Dropout layers active during inference
- Batch normalization in eval mode

**Solutions**:
```python
# ✅ Ensure model is in eval mode
model.eval()

# ✅ Use torch.no_grad()
with torch.no_grad():
    overlay = generate_gradcam_visualization(...)
```

---

## 📚 API Reference

### Main Functions

#### `generate_gradcam_visualization()`

**High-level function for generating GradCAM overlays.**

```python
overlay = generate_gradcam_visualization(
    model,               # PyTorch model
    image_input,         # str (path), PIL Image, numpy array, or torch.Tensor
    target_layer,        # Layer to hook (e.g., model.block4)
    class_idx=None,      # Target class (None = predicted class)
    device='cuda',       # Device
    transform=None,      # Preprocessing transform (auto-detected if None)
    colormap=cv2.COLORMAP_JET,  # OpenCV colormap
    alpha=0.5,           # Overlay transparency (0=original, 1=full heatmap)
    return_heatmap=False # If True, return (overlay, heatmap)
)
```

**Returns**:
- If `return_heatmap=False`: `numpy.ndarray` (H, W, 3) uint8 RGB overlay
- If `return_heatmap=True`: `(overlay, heatmap)` tuple

---

#### `GradCAM` Class

**Low-level class for custom GradCAM generation.**

```python
from utils.gradcam import GradCAM

# Create instance
gradcam = GradCAM(
    model,         # PyTorch model
    target_layer,  # Layer to hook
    device='cuda'  # Device
)

# Generate heatmap
heatmap = gradcam(
    input_tensor,      # torch.Tensor (1, C, H, W)
    class_idx=None,    # Target class (None = predicted)
    retain_graph=False # Retain computation graph
)

# Create overlay
overlay = gradcam.generate_heatmap_overlay(
    original_image,  # numpy.ndarray (H, W, 3)
    heatmap,         # numpy.ndarray (h, w)
    colormap=cv2.COLORMAP_JET,
    alpha=0.5
)
```

---

#### `get_target_layer()`

**Auto-detect last convolutional layer.**

```python
from utils.gradcam import get_target_layer

target_layer = get_target_layer(
    model,       # PyTorch model
    model_type   # 'custom', 'efficientnet_b0', 'resnet18', etc.
)
```

**Returns**: `torch.nn.Module` (last conv layer)

---

#### `visualize_gradcam_grid()`

**Generate GradCAM grid for multiple images.**

```python
from utils.gradcam import visualize_gradcam_grid

grid = visualize_gradcam_grid(
    model,             # PyTorch model
    image_paths,       # List of image paths
    target_layer,      # Layer to hook
    class_indices=None, # List of class indices (None = predicted)
    device='cuda',     # Device
    save_path=None,    # Path to save grid (optional)
    colormap=cv2.COLORMAP_JET,
    alpha=0.5
)
```

**Returns**: `numpy.ndarray` (H, W, 3) grid image

---

#### `get_colormap_options()`

**Get dictionary of available OpenCV colormaps.**

```python
from utils.gradcam import get_colormap_options

colormaps = get_colormap_options()
# Returns: {'jet': cv2.COLORMAP_JET, 'hot': cv2.COLORMAP_HOT, ...}
```

---

## 📖 References

### Research Papers

1. **Grad-CAM: Visual Explanations from Deep Networks**  
   Ramprasaath R. Selvaraju et al., 2017  
   [arXiv:1610.02391](https://arxiv.org/abs/1610.02391)

2. **Learning Deep Features for Discriminative Localization**  
   Bolei Zhou et al., 2016  
   [arXiv:1512.04150](https://arxiv.org/abs/1512.04150)

### Related Documentation

- **PyTorch Hooks**: https://pytorch.org/docs/stable/generated/torch.nn.Module.html#torch.nn.Module.register_forward_hook
- **OpenCV Colormaps**: https://docs.opencv.org/4.x/d3/d50/group__imgproc__colormap.html

---

## ✅ Summary

**You now know how to**:
- ✅ Generate GradCAM visualizations for single images
- ✅ Compare top-k predictions with GradCAM
- ✅ Process batches of images efficiently
- ✅ Interpret heatmap colors and patterns
- ✅ Integrate GradCAM into web applications
- ✅ Troubleshoot common issues
- ✅ Choose appropriate colormaps for different use cases

**Next Steps**:
1. Run `python test_gradcam.py` to verify installation
2. Try `python example_gradcam.py` for usage examples
3. Generate GradCAMs for your trained model
4. Analyze activation patterns to validate model behavior
5. Integrate into your production inference pipeline

---

**Questions?** Check the troubleshooting section or examine `utils/gradcam.py` for implementation details.

**Happy Visualizing! 🎨🔥**
