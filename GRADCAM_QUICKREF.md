# GradCAM Quick Reference

**One-page reference for GradCAM explainability visualization.**

---

## 🚀 Basic Usage (Copy-Paste Ready)

```python
from utils.gradcam import generate_gradcam_visualization
from predict import load_model_once
from PIL import Image

# Load model once
model, class_names, device = load_model_once('models/cropshield_cnn.pth')

# Generate GradCAM
overlay = generate_gradcam_visualization(
    model, 'leaf.jpg', model.block4, device=device
)

# Save
Image.fromarray(overlay).save('gradcam.jpg')
```

---

## 📊 Common Use Cases

### 1. Single Image with Auto-Prediction
```python
overlay = generate_gradcam_visualization(model, 'leaf.jpg', model.block4, device=device)
```

### 2. Specific Class Visualization
```python
overlay = generate_gradcam_visualization(model, 'leaf.jpg', model.block4, class_idx=5, device=device)
```

### 3. Get Raw Heatmap
```python
overlay, heatmap = generate_gradcam_visualization(model, 'leaf.jpg', model.block4, device=device, return_heatmap=True)
```

### 4. Custom Transparency
```python
overlay = generate_gradcam_visualization(model, 'leaf.jpg', model.block4, device=device, alpha=0.3)  # Subtle
overlay = generate_gradcam_visualization(model, 'leaf.jpg', model.block4, device=device, alpha=0.7)  # Strong
```

### 5. Different Colormap
```python
import cv2
overlay = generate_gradcam_visualization(model, 'leaf.jpg', model.block4, device=device, colormap=cv2.COLORMAP_VIRIDIS)
```

---

## 🎯 Target Layers

| Architecture | Target Layer |
|--------------|--------------|
| Custom CNN | `model.block4` |
| EfficientNet | `get_target_layer(model, 'efficientnet_b0')` |
| ResNet | `get_target_layer(model, 'resnet18')` |

```python
from utils.gradcam import get_target_layer

# Auto-detect
target_layer = get_target_layer(model, 'custom')
```

---

## 🎨 Colormaps

| Colormap | Best For | Code |
|----------|----------|------|
| **jet** | General use | `cv2.COLORMAP_JET` |
| **hot** | Medical/thermal | `cv2.COLORMAP_HOT` |
| **viridis** | Publications (colorblind-safe) | `cv2.COLORMAP_VIRIDIS` |
| **plasma** | Presentations | `cv2.COLORMAP_PLASMA` |
| **rainbow** | High detail | `cv2.COLORMAP_RAINBOW` |

```python
from utils.gradcam import get_colormap_options
colormaps = get_colormap_options()  # Dict of all 12 colormaps
```

---

## 🔍 Interpretation

| Color | Meaning |
|-------|---------|
| 🔴 **Red/Hot** | High activation - Model focused here |
| 🟡 **Orange/Warm** | Medium activation - Some attention |
| 🔵 **Blue/Cool** | Low activation - Model ignored this |

### ✅ Good GradCAM
- Red regions on disease symptoms (lesions, spots)
- Blue regions on healthy tissue
- Concentrated, not scattered

### ❌ Bad GradCAM
- Red on background/borders
- Uniform/flat heatmap
- Scattered activation

---

## 🔥 Advanced Features

### Compare Top-3 Predictions
```python
from predict import predict_disease

predictions = predict_disease('leaf.jpg', model, class_names, device, top_k=3)

overlays = []
for class_name, conf in predictions:
    class_idx = class_names.index(class_name)
    overlay = generate_gradcam_visualization(model, 'leaf.jpg', model.block4, class_idx=class_idx, device=device)
    overlays.append(overlay)

# Save side-by-side
import numpy as np
comparison = np.hstack(overlays)
Image.fromarray(comparison).save('topk.jpg')
```

### Batch Grid
```python
from utils.gradcam import visualize_gradcam_grid

grid = visualize_gradcam_grid(
    model, 
    ['leaf1.jpg', 'leaf2.jpg', 'leaf3.jpg', 'leaf4.jpg'],
    model.block4, device=device, save_path='grid.jpg'
)
```

### Low-Level API
```python
from utils.gradcam import GradCAM
from transforms import get_validation_transforms, IMAGENET_MEAN, IMAGENET_STD

# Preprocess
pil_image = Image.open('leaf.jpg')
transform = get_validation_transforms(224, IMAGENET_MEAN, IMAGENET_STD)
image_tensor = transform(pil_image).unsqueeze(0).to(device)

# Create GradCAM
gradcam = GradCAM(model, model.block4, device)

# Generate heatmap
heatmap = gradcam(image_tensor, class_idx=None)  # (14, 14), [0, 1]

# Create overlay
original = np.array(pil_image.resize((224, 224)))
overlay = gradcam.generate_heatmap_overlay(original, heatmap, alpha=0.5)
```

---

## ⚡ Performance

| Device | Time per Image |
|--------|----------------|
| GPU (RTX 4060) | 10-20ms |
| CPU (Intel i7) | 50-100ms |

**Tips**:
- Cache model with `load_model_once()`
- Use GPU when available
- Process in batches for multiple images

---

## 🐛 Quick Troubleshooting

| Problem | Solution |
|---------|----------|
| Uniform heatmap | Use last conv layer (`model.block4`) |
| Background activation | Check dataset quality, retrain |
| CUDA OOM | Use CPU: `device=torch.device('cpu')` |
| Shape mismatch | Ensure image is 224x224 |
| Changing results | Set `model.eval()` |

---

## 📦 Key Functions

```python
from utils.gradcam import (
    generate_gradcam_visualization,  # High-level (use this!)
    GradCAM,                         # Low-level class
    get_target_layer,                # Auto-detect layer
    visualize_gradcam_grid,          # Batch processing
    get_colormap_options,            # Available colormaps
)
```

---

## 🔗 Integration Example (Flask API)

```python
from flask import Flask, request, jsonify, send_file
from predict import load_model_once, predict_disease
from utils.gradcam import generate_gradcam_visualization
import io
from PIL import Image

app = Flask(__name__)
model, class_names, device = load_model_once('models/cropshield_cnn.pth')

@app.route('/predict', methods=['POST'])
def predict():
    file = request.files['image']
    image = Image.open(io.BytesIO(file.read()))
    image.save('temp.jpg')
    
    predictions = predict_disease('temp.jpg', model, class_names, device, top_k=3)
    overlay = generate_gradcam_visualization(model, 'temp.jpg', model.block4, device=device)
    
    Image.fromarray(overlay).save('temp_gradcam.jpg')
    
    return jsonify({
        'predictions': [{'disease': n, 'confidence': float(c)} for n, c in predictions],
        'gradcam_url': '/gradcam/temp_gradcam.jpg'
    })

@app.route('/gradcam/<path:filename>')
def serve(filename):
    return send_file(filename, mimetype='image/jpeg')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
```

---

## 📚 Files

- `utils/gradcam.py` - Main module (700+ lines)
- `test_gradcam.py` - Test suite (9 tests)
- `example_gradcam.py` - Usage examples (8 examples)
- `GRADCAM_GUIDE.md` - Full documentation

---

## ✅ Validation Checklist

- [ ] Run `python test_gradcam.py` (all 9 tests pass)
- [ ] Generate GradCAM for trained model
- [ ] Verify activations on disease symptoms
- [ ] Compare top-k predictions
- [ ] Test different colormaps
- [ ] Integrate into inference pipeline

---

**For full documentation**: See `GRADCAM_GUIDE.md`

**Questions?** Check troubleshooting section or examine `utils/gradcam.py`
