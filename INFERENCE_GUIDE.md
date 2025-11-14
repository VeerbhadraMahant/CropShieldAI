# Fast Inference Guide

## Overview

The `predict.py` module provides lightweight, optimized single-image inference for CropShield AI models, designed for Streamlit and web deployment.

## Key Features

✅ **Fast Inference**: <100ms per image on GPU (typically ~50ms)  
✅ **Model Caching**: Load once, predict many times (no repeated loading)  
✅ **Flexible Input**: Accepts file paths, PIL Images, or numpy arrays  
✅ **Top-K Predictions**: Get top 3 (or any K) most likely classes  
✅ **GradCAM Support**: Optional raw logits return for visualization  
✅ **Batch Processing**: Efficient multi-image inference  
✅ **Streamlit Ready**: Complete integration example included  
✅ **Mixed Precision**: Automatic AMP for 2x speedup on GPU  

## Quick Start

### Basic Usage

```python
from predict import load_model_once, predict_disease

# 1. Load model once (cached for subsequent calls)
model, class_names, device = load_model_once(
    checkpoint_path='checkpoints/best.pth',
    model_type='custom'
)

# 2. Predict on image
predictions = predict_disease(
    image_input='test_image.jpg',
    model=model,
    class_names=class_names,
    device=device,
    top_k=3
)

# 3. Display results
for class_name, prob in predictions:
    print(f"{class_name}: {prob:.2%}")
```

Output:
```
Tomato__late_blight: 94.23%
Tomato__early_blight: 3.45%
Tomato__healthy: 1.12%
```

## API Reference

### `load_model_once()`

Load model from checkpoint with automatic caching.

```python
model, class_names, device = load_model_once(
    checkpoint_path='checkpoints/best.pth',
    model_type='custom',
    force_reload=False,
    device=None
)
```

**Parameters:**
- `checkpoint_path` (str): Path to model checkpoint file
- `model_type` (str): Model architecture (`'custom'` or `'efficientnet_b0'`)
- `force_reload` (bool): Force reload even if cached (default: `False`)
- `device` (torch.device, optional): Device to load model on (auto-detect if `None`)

**Returns:**
- `model` (nn.Module): Loaded model in eval mode
- `class_names` (List[str]): List of class names
- `device` (torch.device): Device model is on

**Features:**
- ✅ **Automatic caching**: First call loads model, subsequent calls return cached instance
- ✅ **Auto device detection**: Uses GPU if available, falls back to CPU
- ✅ **Class inference**: Automatically detects number of classes from checkpoint
- ✅ **Error handling**: Clear error messages for missing/corrupted checkpoints

**Example:**
```python
# First call: loads model (slow)
model, class_names, device = load_model_once('checkpoints/best.pth')
# Output: 📂 Loading model: checkpoints/best.pth
#         ✅ Model loaded: 22 classes

# Subsequent calls: returns cached model (instant)
model, class_names, device = load_model_once('checkpoints/best.pth')
# Output: ✅ Using cached model: checkpoints/best.pth
```

### `predict_disease()`

Fast single-image prediction with top-K results.

```python
predictions = predict_disease(
    image_input='test.jpg',
    model=model,
    class_names=class_names,
    top_k=3,
    device=None,
    return_raw_logits=False,
    use_amp=True
)
```

**Parameters:**
- `image_input` (str | Path | PIL.Image | np.ndarray): Image to predict on
  - File path: `'test.jpg'`
  - PIL Image: `Image.open('test.jpg')`
  - Numpy array: `cv2.imread('test.jpg')`
- `model` (nn.Module): Trained model (must be in eval mode)
- `class_names` (List[str]): List of class names
- `top_k` (int): Number of top predictions (default: `3`)
- `device` (torch.device, optional): Device for inference (auto-detect if `None`)
- `return_raw_logits` (bool): Return raw logits for GradCAM (default: `False`)
- `use_amp` (bool): Use mixed precision for speed (default: `True`)

**Returns:**
- If `return_raw_logits=False`: `List[Tuple[str, float]]`
  - List of `(class_name, probability)` tuples, ordered by probability
  
- If `return_raw_logits=True`: `Tuple[List[Tuple[str, float]], torch.Tensor]`
  - Tuple of `(predictions, logits)` where logits is raw model output for GradCAM

**Features:**
- ✅ **Fast inference**: <100ms target (typically ~50ms on GPU)
- ✅ **Flexible input**: Handles multiple image formats
- ✅ **Automatic preprocessing**: Same transforms as training (resize, crop, normalize)
- ✅ **Mixed precision**: 2x speedup with AMP on GPU
- ✅ **Memory efficient**: Uses `torch.no_grad()` context

**Examples:**

```python
# Basic prediction
predictions = predict_disease('test.jpg', model, class_names, top_k=3)
for class_name, prob in predictions:
    print(f"{class_name}: {prob:.2%}")

# Prediction with PIL Image
from PIL import Image
img = Image.open('test.jpg')
predictions = predict_disease(img, model, class_names, top_k=5)

# Prediction for GradCAM
predictions, logits = predict_disease(
    'test.jpg', model, class_names, 
    return_raw_logits=True
)
# Use logits for GradCAM visualization
```

### `predict_batch()`

Efficient batch inference for multiple images.

```python
results = predict_batch(
    image_list=['img1.jpg', 'img2.jpg', 'img3.jpg'],
    model=model,
    class_names=class_names,
    top_k=3,
    batch_size=32,
    device=None,
    use_amp=True
)
```

**Parameters:**
- `image_list` (List[str | Path | PIL.Image | np.ndarray]): List of images
- `model` (nn.Module): Trained model
- `class_names` (List[str]): Class names
- `top_k` (int): Top predictions per image (default: `3`)
- `batch_size` (int): Batch size for inference (default: `32`)
- `device` (torch.device, optional): Device for inference
- `use_amp` (bool): Use mixed precision (default: `True`)

**Returns:**
- `List[List[Tuple[str, float]]]`: List of predictions for each image

**Example:**
```python
images = ['img1.jpg', 'img2.jpg', 'img3.jpg']
results = predict_batch(images, model, class_names, top_k=3)

for i, predictions in enumerate(results):
    print(f"\nImage {i+1}:")
    for class_name, prob in predictions:
        print(f"  {class_name}: {prob:.2%}")
```

### `clear_model_cache()`

Clear cached models to free memory.

```python
from predict import clear_model_cache

clear_model_cache()
# Output: 🗑️  Model cache cleared
```

## Streamlit Integration

### Complete Streamlit App

Save as `app.py`:

```python
import streamlit as st
from PIL import Image
from predict import load_model_once, predict_disease

# Page configuration
st.set_page_config(
    page_title="CropShield AI",
    page_icon="🌾",
    layout="wide"
)

# Title
st.title("🌾 CropShield AI - Plant Disease Detection")
st.write("Upload a plant image to detect diseases")

# Load model once (cached across reruns)
@st.cache_resource
def load_model():
    """Load model (cached by Streamlit)"""
    model, class_names, device = load_model_once(
        checkpoint_path='checkpoints/best.pth',
        model_type='custom'
    )
    return model, class_names, device

# Load model
with st.spinner("Loading model..."):
    model, class_names, device = load_model()
    st.success("✅ Model loaded!")

# File uploader
uploaded_file = st.file_uploader(
    "Choose an image...",
    type=['jpg', 'jpeg', 'png'],
    help="Upload a clear image of the plant leaf"
)

if uploaded_file is not None:
    # Display image
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📷 Uploaded Image")
        image = Image.open(uploaded_file)
        st.image(image, use_column_width=True)
    
    with col2:
        st.subheader("🔍 Predictions")
        
        # Run prediction
        with st.spinner("Analyzing..."):
            predictions = predict_disease(
                image_input=image,
                model=model,
                class_names=class_names,
                device=device,
                top_k=3
            )
        
        # Display results
        for i, (class_name, prob) in enumerate(predictions, 1):
            # Format class name
            display_name = class_name.replace('_', ' ').title()
            
            # Color based on confidence
            if prob > 0.7:
                color = "🟢"  # High confidence
            elif prob > 0.4:
                color = "🟡"  # Medium confidence
            else:
                color = "🔴"  # Low confidence
            
            # Display prediction
            st.write(f"{color} **{i}. {display_name}**")
            st.progress(prob)
            st.write(f"Confidence: {prob:.1%}")
            st.write("---")
        
        # Show top prediction prominently
        top_class, top_prob = predictions[0]
        
        if top_prob > 0.8:
            st.success(f"✅ **Detected: {top_class.replace('_', ' ').title()}** ({top_prob:.1%})")
        elif top_prob > 0.5:
            st.warning(f"⚠️ **Likely: {top_class.replace('_', ' ').title()}** ({top_prob:.1%})")
        else:
            st.info(f"ℹ️ **Uncertain** - Please provide a clearer image")

# Sidebar
with st.sidebar:
    st.header("ℹ️ About")
    st.write("""
    CropShield AI detects diseases in:
    - 🥔 Potato
    - 🌿 Sugarcane  
    - 🍅 Tomato
    - 🌾 Wheat
    """)
    
    st.metric("Classes", len(class_names))
    st.metric("Device", str(device).upper())
    st.metric("Speed", "~50ms")
```

### Run Streamlit App

```bash
streamlit run app.py
```

Opens browser at `http://localhost:8501`

## CLI Testing

Test inference from command line:

```bash
# Basic test
python predict.py test_image.jpg

# With custom checkpoint
python predict.py test_image.jpg --checkpoint checkpoints/custom_best.pth --model custom

# Show top 5 predictions
python predict.py test_image.jpg --top_k 5

# EfficientNet model
python predict.py test_image.jpg --checkpoint checkpoints/efficientnet_best.pth --model efficientnet_b0
```

Example output:
```
============================================================
🌾 CROPSHIELD AI - INFERENCE TEST
============================================================
Image:       test_image.jpg
Checkpoint:  checkpoints/best.pth
Model:       custom
Top-K:       3
============================================================

📂 Loading model: checkpoints/best.pth
✅ Model loaded: 22 classes
   Device: cuda
   Type: custom

🔍 Running inference...

============================================================
📊 PREDICTIONS (inference: 47.3ms)
============================================================
1. 🟢 Tomato Late Blight                            94.23%  [HIGH]
2. 🟡 Tomato Early Blight                            3.45%  [LOW]
3. 🔴 Tomato Healthy                                 1.12%  [LOW]
============================================================

✅ Excellent latency: 47.3ms (target: <100ms)
```

## Usage Examples

### Example 1: Simple Prediction

```python
from predict import load_model_once, predict_disease

# Load model
model, class_names, device = load_model_once('checkpoints/best.pth')

# Predict
predictions = predict_disease('test.jpg', model, class_names, top_k=3)

# Show results
print(f"Top prediction: {predictions[0][0]} ({predictions[0][1]:.1%})")
```

### Example 2: Process Multiple Images

```python
from predict import load_model_once, predict_batch
from pathlib import Path

# Load model once
model, class_names, device = load_model_once('checkpoints/best.pth')

# Get all images in folder
image_paths = list(Path('test_images').glob('*.jpg'))

# Batch predict (efficient)
results = predict_batch(image_paths, model, class_names, top_k=3, batch_size=32)

# Display results
for img_path, predictions in zip(image_paths, results):
    print(f"\n{img_path.name}:")
    for class_name, prob in predictions:
        print(f"  {class_name}: {prob:.2%}")
```

### Example 3: Integration with OpenCV

```python
import cv2
from predict import load_model_once, predict_disease

# Load model
model, class_names, device = load_model_once('checkpoints/best.pth')

# Read image with OpenCV
image = cv2.imread('test.jpg')
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Predict (handles numpy arrays)
predictions = predict_disease(image_rgb, model, class_names, top_k=3)

# Display on image
for i, (class_name, prob) in enumerate(predictions):
    text = f"{i+1}. {class_name}: {prob:.1%}"
    cv2.putText(image, text, (10, 30 + i*30), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

cv2.imshow('Predictions', image)
cv2.waitKey(0)
```

### Example 4: GradCAM Integration

```python
from predict import load_model_once, predict_disease

# Load model
model, class_names, device = load_model_once('checkpoints/best.pth')

# Predict with raw logits for GradCAM
predictions, logits = predict_disease(
    'test.jpg', model, class_names,
    return_raw_logits=True
)

# Get predicted class index
predicted_class = class_names.index(predictions[0][0])

# Use logits for GradCAM backward pass
logits[0, predicted_class].backward()
# ... GradCAM visualization code ...
```

### Example 5: Real-time Webcam

```python
import cv2
from predict import load_model_once, predict_disease

# Load model
model, class_names, device = load_model_once('checkpoints/best.pth')

# Open webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    # Predict every 10 frames (for speed)
    if cap.get(cv2.CAP_PROP_POS_FRAMES) % 10 == 0:
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        predictions = predict_disease(frame_rgb, model, class_names, top_k=1)
        
        # Display top prediction
        class_name, prob = predictions[0]
        text = f"{class_name}: {prob:.1%}"
        cv2.putText(frame, text, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    
    cv2.imshow('CropShield AI', frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
```

## Performance Optimization

### GPU vs CPU

```python
# Force GPU
model, class_names, device = load_model_once(
    'checkpoints/best.pth',
    device=torch.device('cuda')
)

# Force CPU
model, class_names, device = load_model_once(
    'checkpoints/best.pth',
    device=torch.device('cpu')
)
```

**Typical Performance:**
- GPU (RTX 4060): ~50ms per image
- CPU (Intel i7): ~200ms per image

### Batch Size Tuning

```python
# Small batch (memory constrained)
results = predict_batch(images, model, class_names, batch_size=16)

# Large batch (faster)
results = predict_batch(images, model, class_names, batch_size=64)
```

### Disable AMP (for debugging)

```python
predictions = predict_disease(
    'test.jpg', model, class_names,
    use_amp=False  # Disable mixed precision
)
```

## Image Preprocessing

The inference pipeline uses the same preprocessing as training:

1. **Resize**: Shorter side to 256 pixels
2. **Center Crop**: 224×224 pixels
3. **To Tensor**: Convert to [0, 1] float tensor
4. **Normalize**: ImageNet mean/std
   - Mean: `[0.485, 0.456, 0.406]`
   - Std: `[0.229, 0.224, 0.225]`

**Supported Input Formats:**
- RGB images (preferred)
- RGBA images (converted to RGB)
- Grayscale images (converted to RGB)
- Numpy arrays (uint8 or float32)
- PIL Images
- File paths (str or Path)

## Troubleshooting

### Issue 1: Slow Inference (>100ms)

**Symptoms**: Predictions taking longer than expected

**Solutions:**
1. Ensure GPU is being used:
   ```python
   print(f"Device: {device}")  # Should be 'cuda'
   ```

2. Check AMP is enabled:
   ```python
   predictions = predict_disease(..., use_amp=True)
   ```

3. Warm up model (first inference is slower):
   ```python
   # Dummy inference for warmup
   dummy = torch.randn(1, 3, 224, 224).to(device)
   model(dummy)
   # Now real inferences will be faster
   ```

### Issue 2: Out of Memory

**Symptoms**: `CUDA out of memory` error

**Solutions:**
1. Use CPU inference:
   ```python
   model, class_names, device = load_model_once(
       'checkpoints/best.pth',
       device=torch.device('cpu')
   )
   ```

2. Reduce batch size:
   ```python
   results = predict_batch(..., batch_size=8)
   ```

### Issue 3: Model Not Loading

**Symptoms**: `FileNotFoundError` or `RuntimeError`

**Solutions:**
1. Verify checkpoint path:
   ```bash
   ls checkpoints/
   ```

2. Ensure model type matches checkpoint:
   ```python
   # If checkpoint is from custom CNN
   model, _, _ = load_model_once('checkpoints/best.pth', model_type='custom')
   
   # If checkpoint is from EfficientNet
   model, _, _ = load_model_once('checkpoints/best.pth', model_type='efficientnet_b0')
   ```

### Issue 4: Wrong Predictions

**Symptoms**: Predictions don't match expected results

**Solutions:**
1. Verify model is in eval mode:
   ```python
   model.eval()  # Should be set automatically
   ```

2. Check image preprocessing:
   ```python
   from predict import get_inference_transform
   transform = get_inference_transform()
   # Verify transform matches training
   ```

3. Validate checkpoint:
   ```bash
   # Test with evaluation script first
   python evaluate.py --model custom --checkpoint checkpoints/best.pth
   ```

## Best Practices

1. **Load model once** at application startup
   ```python
   # Good: Load once, predict many
   model, class_names, device = load_model_once('checkpoints/best.pth')
   for img in images:
       predictions = predict_disease(img, model, class_names, device=device)
   
   # Bad: Loading model repeatedly
   for img in images:
       model, class_names, device = load_model_once('checkpoints/best.pth')  # Slow!
       predictions = predict_disease(img, model, class_names, device=device)
   ```

2. **Use batch inference** for multiple images
   ```python
   # Good: Batch inference
   results = predict_batch(images, model, class_names, batch_size=32)
   
   # Less efficient: Individual predictions
   results = [predict_disease(img, model, class_names) for img in images]
   ```

3. **Cache model in Streamlit**
   ```python
   @st.cache_resource
   def load_model():
       return load_model_once('checkpoints/best.pth')
   ```

4. **Warm up model** for latency-critical apps
   ```python
   # One-time warmup
   dummy = torch.randn(1, 3, 224, 224).to(device)
   model(dummy)
   ```

5. **Handle errors gracefully**
   ```python
   try:
       predictions = predict_disease('test.jpg', model, class_names)
   except FileNotFoundError:
       print("Image not found")
   except RuntimeError:
       print("Inference failed")
   ```

## Summary

The `predict.py` module provides:

- ✅ **Fast inference** (<100ms target, typically ~50ms)
- ✅ **Model caching** (load once, predict many times)
- ✅ **Flexible input** (paths, PIL, numpy)
- ✅ **Top-K predictions** (configurable)
- ✅ **GradCAM support** (optional raw logits)
- ✅ **Batch processing** (efficient multi-image)
- ✅ **Streamlit ready** (complete example)
- ✅ **Production ready** (error handling, logging)

Perfect for deployment in web apps, Streamlit dashboards, or real-time applications! 🚀
