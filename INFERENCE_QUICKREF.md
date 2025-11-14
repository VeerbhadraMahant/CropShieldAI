# Inference Quick Reference

## 🚀 Quick Start

```python
from predict import load_model_once, predict_disease

# Load model (cached)
model, class_names, device = load_model_once('checkpoints/best.pth')

# Predict
predictions = predict_disease('test.jpg', model, class_names, top_k=3)

# Display
for class_name, prob in predictions:
    print(f"{class_name}: {prob:.2%}")
```

## 📚 Core Functions

### `load_model_once()`

Load model with automatic caching:

```python
model, class_names, device = load_model_once(
    checkpoint_path='checkpoints/best.pth',
    model_type='custom',           # or 'efficientnet_b0'
    force_reload=False,             # Reload even if cached
    device=None                     # Auto-detect GPU/CPU
)
```

✅ First call: Loads model  
✅ Subsequent calls: Returns cached (instant)

### `predict_disease()`

Fast single-image prediction:

```python
predictions = predict_disease(
    image_input='test.jpg',        # Path, PIL Image, or numpy array
    model=model,
    class_names=class_names,
    top_k=3,                        # Top predictions
    device=None,                    # Auto-detect
    return_raw_logits=False,        # For GradCAM
    use_amp=True                    # Mixed precision (2x speedup)
)
```

Returns: `[(class_name, probability), ...]`

### `predict_batch()`

Efficient multi-image inference:

```python
results = predict_batch(
    image_list=['img1.jpg', 'img2.jpg'],
    model=model,
    class_names=class_names,
    top_k=3,
    batch_size=32
)
```

Returns: `[[(class, prob), ...], ...]` for each image

## 🎨 Streamlit Template

Save as `app.py`:

```python
import streamlit as st
from PIL import Image
from predict import load_model_once, predict_disease

st.set_page_config(page_title="CropShield AI", page_icon="🌾")
st.title("🌾 CropShield AI")

# Load model (cached)
@st.cache_resource
def load_model():
    return load_model_once('checkpoints/best.pth', model_type='custom')

model, class_names, device = load_model()

# Upload
uploaded_file = st.file_uploader("Choose an image", type=['jpg', 'png'])

if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image, width=400)
    
    # Predict
    predictions = predict_disease(image, model, class_names, device=device, top_k=3)
    
    # Display
    for i, (class_name, prob) in enumerate(predictions, 1):
        st.write(f"{i}. **{class_name}**: {prob:.1%}")
        st.progress(prob)
```

Run: `streamlit run app.py`

## ⚡ CLI Testing

```bash
# Basic test
python predict.py test_image.jpg

# Custom checkpoint
python predict.py test_image.jpg --checkpoint checkpoints/custom_best.pth

# Top 5 predictions
python predict.py test_image.jpg --top_k 5

# EfficientNet
python predict.py test_image.jpg --model efficientnet_b0 --checkpoint checkpoints/efficientnet_best.pth
```

## 🔧 Input Formats

All accepted:

```python
# File path (str or Path)
predict_disease('test.jpg', model, class_names)

# PIL Image
from PIL import Image
img = Image.open('test.jpg')
predict_disease(img, model, class_names)

# Numpy array (OpenCV)
import cv2
img = cv2.imread('test.jpg')
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
predict_disease(img_rgb, model, class_names)

# Streamlit uploaded file
uploaded = st.file_uploader("Image", type=['jpg'])
img = Image.open(uploaded)
predict_disease(img, model, class_names)
```

## 📊 Common Patterns

### Pattern 1: Process Folder

```python
from pathlib import Path

model, class_names, device = load_model_once('checkpoints/best.pth')

for img_path in Path('test_images').glob('*.jpg'):
    predictions = predict_disease(img_path, model, class_names, top_k=1)
    print(f"{img_path.name}: {predictions[0][0]}")
```

### Pattern 2: Batch Processing

```python
images = ['img1.jpg', 'img2.jpg', 'img3.jpg']
results = predict_batch(images, model, class_names, batch_size=32)

for img, preds in zip(images, results):
    print(f"\n{img}:")
    for class_name, prob in preds:
        print(f"  {class_name}: {prob:.2%}")
```

### Pattern 3: GradCAM

```python
# Get predictions + logits
predictions, logits = predict_disease(
    'test.jpg', model, class_names,
    return_raw_logits=True
)

# Use logits for visualization
predicted_idx = class_names.index(predictions[0][0])
logits[0, predicted_idx].backward()
# ... GradCAM code ...
```

### Pattern 4: Confidence Filtering

```python
predictions = predict_disease('test.jpg', model, class_names, top_k=5)

# Filter by confidence threshold
high_confidence = [(c, p) for c, p in predictions if p > 0.5]

if high_confidence:
    print(f"Confident prediction: {high_confidence[0][0]}")
else:
    print("Low confidence - image unclear")
```

## 🎯 Performance

| Metric | GPU (RTX 4060) | CPU (i7) |
|--------|----------------|----------|
| Single Image | ~50ms | ~200ms |
| Batch (32 images) | ~800ms | ~6s |
| With AMP | 2x faster | N/A |

**Target**: <100ms per image on GPU ✅

## 🔍 Preprocessing Pipeline

Same as training:

1. Resize → 256px (shorter side)
2. Center Crop → 224×224
3. To Tensor → [0, 1]
4. Normalize → ImageNet mean/std

## 🐛 Troubleshooting

| Issue | Solution |
|-------|----------|
| Slow inference (>100ms) | Check GPU usage, enable AMP |
| CUDA OOM | Reduce batch size or use CPU |
| Model not loading | Verify checkpoint path and model type |
| Wrong predictions | Ensure model in eval mode, check preprocessing |

## 💡 Best Practices

1. **Load once, predict many**
   ```python
   model, class_names, device = load_model_once('checkpoints/best.pth')
   for img in images:
       predict_disease(img, model, class_names, device=device)
   ```

2. **Use batch for multiple images**
   ```python
   results = predict_batch(images, model, class_names, batch_size=32)
   ```

3. **Cache in Streamlit**
   ```python
   @st.cache_resource
   def load_model():
       return load_model_once('checkpoints/best.pth')
   ```

4. **Warm up for latency-critical apps**
   ```python
   dummy = torch.randn(1, 3, 224, 224).to(device)
   model(dummy)  # Warmup
   ```

## 📦 Function Signatures

```python
# Load model (cached)
load_model_once(
    checkpoint_path: str,
    model_type: str = 'custom',
    force_reload: bool = False,
    device: Optional[torch.device] = None
) -> Tuple[nn.Module, List[str], torch.device]

# Single prediction
predict_disease(
    image_input: Union[str, Path, Image.Image, np.ndarray],
    model: nn.Module,
    class_names: List[str],
    top_k: int = 3,
    device: Optional[torch.device] = None,
    return_raw_logits: bool = False,
    use_amp: bool = True
) -> Union[List[Tuple[str, float]], Tuple[List[Tuple[str, float]], torch.Tensor]]

# Batch prediction
predict_batch(
    image_list: List[Union[str, Path, Image.Image, np.ndarray]],
    model: nn.Module,
    class_names: List[str],
    top_k: int = 3,
    batch_size: int = 32,
    device: Optional[torch.device] = None,
    use_amp: bool = True
) -> List[List[Tuple[str, float]]]

# Clear cache
clear_model_cache() -> None
```

## 🎓 Full Documentation

See `INFERENCE_GUIDE.md` for:
- Complete API reference
- Advanced examples
- Streamlit integration
- OpenCV integration
- Real-time webcam
- Performance optimization
- Detailed troubleshooting

---

**Fast, cached, production-ready inference! 🚀**
