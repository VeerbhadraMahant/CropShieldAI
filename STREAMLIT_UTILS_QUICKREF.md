# Streamlit App Utilities - Quick Reference

**Complete guide for using `utils/app_utils.py` in Streamlit applications.**

---

## 📋 Table of Contents

1. [Class Mapping](#class-mapping)
2. [Image Handling](#image-handling)
3. [Prediction Visualization](#prediction-visualization)
4. [GradCAM Visualization](#gradcam-visualization)
5. [Utility Functions](#utility-functions)
6. [Complete Example](#complete-example)

---

## 🗂️ Class Mapping

### Load Class Names from JSON

```python
from utils.app_utils import load_class_names

# Load class names
class_names = load_class_names('models/class_to_idx.json')

# Use in app
st.selectbox("Select Disease", class_names)
```

**File format** (`class_to_idx.json`):
```json
{
  "Potato__early_blight": 0,
  "Potato__healthy": 1,
  "Potato__late_blight": 2
}
```

### Load Full Mapping

```python
from utils.app_utils import load_class_mapping

# Get dict mapping
mapping = load_class_mapping('models/class_to_idx.json')

# Get index for class
idx = mapping['Potato__early_blight']  # 0
```

---

## 🖼️ Image Handling

### Convert Uploaded File to PIL

```python
from utils.app_utils import uploaded_file_to_pil

# In Streamlit
uploaded_file = st.file_uploader("Upload Image", type=['jpg', 'png'])

# Convert to PIL
image = uploaded_file_to_pil(uploaded_file)

if image:
    st.image(image, caption="Uploaded Image")
```

### Convert BytesIO to PIL

```python
from utils.app_utils import bytesio_to_pil
import io

# If you have BytesIO object
image_bytes = uploaded_file.getvalue()
bytesio = io.BytesIO(image_bytes)

# Convert
image = bytesio_to_pil(bytesio)
```

### Resize Image

```python
from utils.app_utils import resize_image

# Resize for display or inference
small_image = resize_image(image, max_size=224, maintain_aspect=True)

st.image(small_image, width=224)
```

### Convert PIL to Bytes

```python
from utils.app_utils import pil_to_bytes

# Convert to bytes
image_bytes = pil_to_bytes(image, format='PNG')

# Use for download button
st.download_button(
    label="Download Image",
    data=image_bytes,
    file_name="output.png",
    mime="image/png"
)
```

---

## 📊 Prediction Visualization

### Display Predictions (Plotly)

```python
from utils.app_utils import display_predictions

# Your predictions
predictions = [
    ('Potato__late_blight', 0.87),
    ('Potato__early_blight', 0.09),
    ('Potato__healthy', 0.04)
]

# Display interactive chart
display_predictions(
    predictions,
    title="Top 3 Disease Predictions",
    use_plotly=True,
    height=400,
    color_scale="viridis"
)
```

**Available color scales**:
- `viridis`, `plasma`, `inferno`, `magma`
- `blues`, `greens`, `reds`, `purples`
- `rainbow`, `jet`, `hot`, `cool`

### Display Predictions (Matplotlib)

```python
# Use Matplotlib instead
display_predictions(
    predictions,
    use_plotly=False,  # Use Matplotlib
    height=400,
    color_scale="viridis"
)
```

### HTML Table Format

```python
from utils.app_utils import format_prediction_table

# Create HTML table
html = format_prediction_table(predictions)

# Display in Streamlit
st.markdown(html, unsafe_allow_html=True)
```

---

## 🔥 GradCAM Visualization

### Side-by-Side Comparison

```python
from utils.app_utils import show_gradcam_overlay
from utils.gradcam import generate_gradcam_visualization
from predict import load_model_once

# Load model
model, class_names, device = load_model_once('models/cropshield_cnn.pth')

# Generate GradCAM
overlay = generate_gradcam_visualization(
    model, 'leaf.jpg', model.block4, device=device
)

# Display side-by-side
show_gradcam_overlay(
    original=image,
    heatmap=overlay,
    column_width=(1, 1),  # Equal columns
    titles=("📷 Original", "🔥 GradCAM"),
    captions=(
        "Input image",
        "Red = High attention, Blue = Low attention"
    )
)
```

### Grid Layout

```python
from utils.app_utils import show_gradcam_grid

# Multiple overlays
overlays = [overlay1, overlay2, overlay3, overlay4]
titles = ["Top-1", "Top-2", "Top-3", "Top-4"]

# Display in grid
show_gradcam_grid(
    images=overlays,
    titles=titles,
    columns=2  # 2x2 grid
)
```

### Compare Top-K Predictions

```python
from utils.app_utils import show_gradcam_comparison

# Generate overlays for top-3
overlays = []
for cls, conf in predictions[:3]:
    class_idx = class_names.index(cls)
    overlay = generate_gradcam_visualization(
        model, image_path, model.block4,
        class_idx=class_idx, device=device
    )
    overlays.append(overlay)

# Show comparison
show_gradcam_comparison(
    original=image,
    overlays=overlays,
    predictions=predictions[:3]
)
```

---

## 🛠️ Utility Functions

### Confidence Indicator

```python
from utils.app_utils import create_confidence_indicator

# Get top prediction confidence
top_conf = predictions[0][1]

# Create colored badge
html = create_confidence_indicator(
    confidence=top_conf,
    thresholds=(0.7, 0.4)  # (high, medium) thresholds
)

st.markdown(html, unsafe_allow_html=True)
```

**Output**:
- ✅ **Green** (>70%): High Confidence
- ⚠️ **Orange** (40-70%): Medium Confidence
- ❌ **Red** (<40%): Low Confidence

### Save to History

```python
from utils.app_utils import save_prediction_history

# Create history entry
entry = save_prediction_history(
    image=image,
    predictions=predictions,
    metadata={
        'crop': 'Potato',
        'location': 'Farm A',
        'weather': {'temp': 25, 'humidity': 60}
    }
)

# Store in session state
if 'history' not in st.session_state:
    st.session_state['history'] = []

st.session_state['history'].append(entry)
```

**Entry format**:
```python
{
    'timestamp': '2025-11-10T10:30:00',
    'predictions': [('Disease', 0.87), ...],
    'image_size': (224, 224),
    'image_mode': 'RGB',
    'metadata': {'crop': 'Potato', ...}
}
```

### Crop Emoji

```python
from utils.app_utils import get_crop_emoji

# Get emoji for crop
emoji = get_crop_emoji('Potato')  # 🥔
emoji = get_crop_emoji('Tomato')  # 🍅
emoji = get_crop_emoji('Wheat')   # 🌾

# Use in UI
st.markdown(f"### {emoji} {crop_name}")
```

**Supported crops**:
- 🥔 Potato
- 🍅 Tomato
- 🌾 Wheat
- 🌾 Rice
- 🌽 Corn
- 🎋 Sugarcane
- ☁️ Cotton
- 🍎 Apple
- 🍇 Grape
- 🍊 Orange
- 🍌 Banana
- 🌱 Default (any other)

---

## 📝 Complete Example

### Full Streamlit App

```python
import streamlit as st
from utils.app_utils import (
    uploaded_file_to_pil,
    display_predictions,
    show_gradcam_overlay,
    create_confidence_indicator,
    get_crop_emoji,
    save_prediction_history
)
from predict import load_model_once, predict_disease
from utils.gradcam import generate_gradcam_visualization

# Page config
st.set_page_config(
    page_title="CropShield AI",
    page_icon="🌱",
    layout="wide"
)

# Initialize session state
if 'history' not in st.session_state:
    st.session_state['history'] = []

# Header
st.title("🌱 CropShield AI - Disease Detection")

# Sidebar
with st.sidebar:
    crop_type = st.selectbox("Crop Type", ["Potato", "Tomato", "Wheat"])
    crop_emoji = get_crop_emoji(crop_type)
    st.markdown(f"### {crop_emoji} {crop_type}")

# Main content
col1, col2 = st.columns(2)

with col1:
    # Upload
    st.markdown("### 📤 Upload Image")
    uploaded = st.file_uploader("Choose image", type=['jpg', 'png'])
    
    if uploaded:
        image = uploaded_file_to_pil(uploaded)
        st.image(image, use_container_width=True)

with col2:
    # Results
    st.markdown("### 🔬 Results")
    
    if uploaded and image:
        if st.button("Analyze", type="primary"):
            with st.spinner("Analyzing..."):
                # Load model (cached)
                model, class_names, device = load_model_once('models/cropshield_cnn.pth')
                
                # Save temp
                import tempfile
                with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as tmp:
                    image.save(tmp.name)
                    
                    # Predict
                    predictions = predict_disease(
                        tmp.name, model, class_names, device, top_k=3
                    )
                    
                    # Store
                    st.session_state['predictions'] = predictions
                    st.session_state['temp_path'] = tmp.name
                
                # Save to history
                entry = save_prediction_history(
                    image, predictions, {'crop': crop_type}
                )
                st.session_state['history'].append(entry)

# Display results
if 'predictions' in st.session_state:
    predictions = st.session_state['predictions']
    
    # Top prediction
    top_disease, top_conf = predictions[0]
    st.markdown(f"### {top_disease.replace('_', ' ').title()}")
    
    # Confidence
    conf_html = create_confidence_indicator(top_conf)
    st.markdown(conf_html, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Chart
    st.markdown("### 📊 Detailed Predictions")
    display_predictions(predictions, use_plotly=True, height=400)
    
    st.markdown("---")
    
    # GradCAM
    st.markdown("### 🔥 Model Explainability")
    
    with st.spinner("Generating GradCAM..."):
        overlay = generate_gradcam_visualization(
            model,
            st.session_state['temp_path'],
            model.block4,
            device=device
        )
        
        show_gradcam_overlay(
            original=image,
            heatmap=overlay,
            titles=("📷 Original", "🔥 Model Focus")
        )

# History
if st.session_state['history']:
    st.markdown("---")
    st.markdown("### 📜 History")
    st.info(f"Total: {len(st.session_state['history'])} analyses")
```

---

## 🎨 Styling Tips

### Custom Column Widths

```python
# 2:1 ratio (left column twice as wide)
col1, col2 = st.columns([2, 1])

with col1:
    st.image(image, use_container_width=True)

with col2:
    display_predictions(predictions)
```

### Tabs Layout

```python
tab1, tab2, tab3 = st.tabs(["📊 Predictions", "🔥 GradCAM", "📜 History"])

with tab1:
    display_predictions(predictions)

with tab2:
    show_gradcam_overlay(original, overlay)

with tab3:
    for entry in st.session_state['history']:
        st.json(entry)
```

### Expanders

```python
with st.expander("ℹ️ How to interpret results"):
    st.markdown("""
    **Confidence Levels:**
    - ✅ High (>70%): Very confident
    - ⚠️ Medium (40-70%): Moderate confidence
    - ❌ Low (<40%): Low confidence
    """)
```

---

## 🐛 Troubleshooting

### Issue: Plotly not available

**Solution**: Install Plotly
```bash
pip install plotly
```

Or use Matplotlib:
```python
display_predictions(predictions, use_plotly=False)
```

### Issue: Image conversion fails

**Solution**: Check image format
```python
image = uploaded_file_to_pil(uploaded)

if image is None:
    st.error("Failed to load image. Please try a different file.")
else:
    st.success(f"Image loaded: {image.size}")
```

### Issue: GradCAM not showing

**Solution**: Verify model and layer
```python
try:
    overlay = generate_gradcam_visualization(model, image_path, ...)
    show_gradcam_overlay(original, overlay)
except Exception as e:
    st.error(f"GradCAM failed: {e}")
    st.info("Make sure model is loaded and image path is correct")
```

---

## 📚 API Reference

### Function Signatures

```python
# Class mapping
load_class_names(path: str | Path) -> List[str]
load_class_mapping(path: str | Path) -> Dict[str, int]

# Image handling
bytesio_to_pil(bytesio_obj: BytesIO) -> Image.Image
uploaded_file_to_pil(uploaded_file) -> Optional[Image.Image]
pil_to_bytes(image: Image.Image, format: str = 'PNG') -> bytes
resize_image(image: Image.Image, max_size: int = 224, maintain_aspect: bool = True) -> Image.Image

# Prediction visualization
display_predictions(
    predictions: List[Tuple[str, float]],
    title: str = "Top Predictions",
    use_plotly: bool = True,
    height: int = 400,
    color_scale: str = "viridis"
) -> None

format_prediction_table(predictions: List[Tuple[str, float]]) -> str

# GradCAM visualization
show_gradcam_overlay(
    original: Image.Image | np.ndarray,
    heatmap: Image.Image | np.ndarray,
    column_width: Optional[Tuple[int, int]] = None,
    titles: Optional[Tuple[str, str]] = None,
    captions: Optional[Tuple[str, str]] = None
) -> None

show_gradcam_grid(
    images: List[Image.Image | np.ndarray],
    titles: Optional[List[str]] = None,
    columns: int = 3
) -> None

show_gradcam_comparison(
    original: Image.Image | np.ndarray,
    overlays: List[Image.Image | np.ndarray],
    predictions: List[Tuple[str, float]]
) -> None

# Utilities
create_confidence_indicator(
    confidence: float,
    thresholds: Tuple[float, float] = (0.7, 0.4)
) -> str

save_prediction_history(
    image: Image.Image,
    predictions: List[Tuple[str, float]],
    metadata: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]

get_crop_emoji(crop_name: str) -> str
```

---

## ✅ Best Practices

1. **Cache model loading**: Use `@st.cache_resource` for model loading
2. **Handle errors gracefully**: Wrap operations in try-except with user-friendly messages
3. **Provide feedback**: Use spinners, progress bars, and status messages
4. **Clean up temp files**: Delete temporary files after use
5. **Save history**: Keep track of analyses for user reference
6. **Responsive layout**: Use columns and tabs for better organization
7. **Accessibility**: Provide captions and help text

---

## 🚀 Performance Tips

- **Lazy loading**: Only load model when needed
- **Image resizing**: Resize large images before processing
- **Batch processing**: Process multiple images together when possible
- **Caching**: Use `@st.cache_resource` and `@st.cache_data`
- **Session state**: Store results in session state to avoid recomputation

---

**For full example**: See `example_streamlit_with_utils.py`

**Questions?** Check the docstrings in `utils/app_utils.py`
