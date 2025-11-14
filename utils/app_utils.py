"""
Streamlit App Utilities
========================

Reusable helper functions for CropShield AI Streamlit application.

This module provides:
- Class mapping utilities
- Prediction visualization
- Image handling for uploads
- GradCAM overlay display
- Confidence chart generation

Author: CropShield AI Team
"""

import json
import io
from pathlib import Path
from typing import Dict, List, Tuple, Union, Optional, Any

import numpy as np
from PIL import Image
import streamlit as st

# Optional imports with fallbacks
try:
    import plotly.graph_objects as go
    import plotly.express as px
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False

try:
    import matplotlib.pyplot as plt
    import matplotlib
    matplotlib.use('Agg')  # Non-interactive backend for Streamlit
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False


# ============================================================================
# Class Mapping Utilities
# ============================================================================

def load_class_names(path: Union[str, Path]) -> List[str]:
    """
    Load class names from a JSON mapping file.
    
    Args:
        path: Path to JSON file containing class_to_idx mapping
              Format: {"class_name": index, ...}
    
    Returns:
        List of class names sorted by index
        
    Raises:
        FileNotFoundError: If JSON file doesn't exist
        ValueError: If JSON format is invalid
        
    Example:
        >>> class_names = load_class_names('models/class_to_idx.json')
        >>> print(class_names[0])
        'Potato__early_blight'
    """
    path = Path(path)
    
    if not path.exists():
        raise FileNotFoundError(f"Class mapping file not found: {path}")
    
    try:
        with open(path, 'r', encoding='utf-8') as f:
            class_to_idx = json.load(f)
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON format in {path}: {e}")
    
    if not isinstance(class_to_idx, dict):
        raise ValueError(f"Expected dict in JSON file, got {type(class_to_idx)}")
    
    # Convert to list sorted by index
    try:
        idx_to_class = {int(idx): name for name, idx in class_to_idx.items()}
        max_idx = max(idx_to_class.keys())
        class_names = [idx_to_class.get(i, f"Unknown_{i}") for i in range(max_idx + 1)]
    except (ValueError, TypeError) as e:
        raise ValueError(f"Invalid index values in JSON: {e}")
    
    return class_names


def load_class_mapping(path: Union[str, Path]) -> Dict[str, int]:
    """
    Load class-to-index mapping from JSON file.
    
    Args:
        path: Path to JSON file
    
    Returns:
        Dictionary mapping class names to indices
        
    Example:
        >>> mapping = load_class_mapping('models/class_to_idx.json')
        >>> mapping['Potato__early_blight']
        0
    """
    path = Path(path)
    
    if not path.exists():
        raise FileNotFoundError(f"Class mapping file not found: {path}")
    
    with open(path, 'r', encoding='utf-8') as f:
        class_to_idx = json.load(f)
    
    return class_to_idx


# ============================================================================
# Image Handling Utilities
# ============================================================================

def bytesio_to_pil(bytesio_obj: io.BytesIO) -> Image.Image:
    """
    Convert BytesIO object (from Streamlit file uploader) to PIL Image.
    
    Args:
        bytesio_obj: BytesIO object containing image data
    
    Returns:
        PIL Image object in RGB mode
        
    Raises:
        ValueError: If image cannot be opened
        
    Example:
        >>> uploaded_file = st.file_uploader("Upload image")
        >>> if uploaded_file:
        ...     pil_image = bytesio_to_pil(uploaded_file)
        ...     st.image(pil_image)
    """
    try:
        # Reset pointer to beginning
        bytesio_obj.seek(0)
        
        # Open image
        image = Image.open(bytesio_obj)
        
        # Convert to RGB (handle RGBA, grayscale, etc.)
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        return image
    
    except Exception as e:
        raise ValueError(f"Failed to convert BytesIO to PIL Image: {e}")


def uploaded_file_to_pil(uploaded_file) -> Optional[Image.Image]:
    """
    Convert Streamlit UploadedFile to PIL Image.
    
    Args:
        uploaded_file: Streamlit UploadedFile object (or None)
    
    Returns:
        PIL Image in RGB mode, or None if no file uploaded
        
    Example:
        >>> uploaded = st.file_uploader("Upload", type=['jpg', 'png'])
        >>> image = uploaded_file_to_pil(uploaded)
        >>> if image:
        ...     st.image(image)
    """
    if uploaded_file is None:
        return None
    
    try:
        # Get bytes
        image_bytes = uploaded_file.getvalue()
        
        # Convert to PIL
        bytesio = io.BytesIO(image_bytes)
        return bytesio_to_pil(bytesio)
    
    except Exception as e:
        st.error(f"Failed to load image: {e}")
        return None


def pil_to_bytes(image: Image.Image, format: str = 'PNG') -> bytes:
    """
    Convert PIL Image to bytes.
    
    Args:
        image: PIL Image object
        format: Output format ('PNG', 'JPEG', etc.)
    
    Returns:
        Image as bytes
        
    Example:
        >>> image = Image.open('leaf.jpg')
        >>> image_bytes = pil_to_bytes(image)
    """
    buffer = io.BytesIO()
    image.save(buffer, format=format)
    buffer.seek(0)
    return buffer.getvalue()


def resize_image(
    image: Image.Image, 
    max_size: int = 224, 
    maintain_aspect: bool = True
) -> Image.Image:
    """
    Resize image for display or inference.
    
    Args:
        image: PIL Image
        max_size: Maximum dimension (width or height)
        maintain_aspect: If True, maintain aspect ratio
    
    Returns:
        Resized PIL Image
        
    Example:
        >>> image = Image.open('large_image.jpg')
        >>> small = resize_image(image, max_size=224)
    """
    if maintain_aspect:
        # Calculate new size maintaining aspect ratio
        width, height = image.size
        
        # Scale based on the larger dimension
        if width >= height:
            new_width = max_size
            new_height = int((height / width) * max_size)
        else:
            new_height = max_size
            new_width = int((width / height) * max_size)
        
        # Ensure minimum size of 1
        new_width = max(1, new_width)
        new_height = max(1, new_height)
    else:
        new_width = new_height = max_size
    
    return image.resize((new_width, new_height), Image.Resampling.LANCZOS)


# ============================================================================
# Prediction Visualization
# ============================================================================

def display_predictions(
    predictions: List[Tuple[str, float]],
    title: str = "Top Predictions",
    use_plotly: bool = True,
    height: int = 400,
    color_scale: str = "viridis"
) -> None:
    """
    Display prediction results as interactive bar chart.
    
    Args:
        predictions: List of (class_name, confidence) tuples
        title: Chart title
        use_plotly: If True use Plotly, else Matplotlib
        height: Chart height in pixels
        color_scale: Color scheme ('viridis', 'plasma', 'blues', etc.)
    
    Example:
        >>> predictions = [
        ...     ('Potato__late_blight', 0.87),
        ...     ('Potato__early_blight', 0.09),
        ...     ('Potato__healthy', 0.04)
        ... ]
        >>> display_predictions(predictions)
    """
    if not predictions:
        st.warning("⚠️ No predictions to display")
        return
    
    # Prepare data
    classes = [pred[0] for pred in predictions]
    confidences = [pred[1] * 100 for pred in predictions]  # Convert to percentage
    
    # Clean class names (remove prefixes like "Potato__")
    display_names = []
    for cls in classes:
        # Remove crop prefix if present
        if '__' in cls:
            parts = cls.split('__')
            display_name = parts[-1].replace('_', ' ').title()
        else:
            display_name = cls.replace('_', ' ').title()
        display_names.append(display_name)
    
    if use_plotly and PLOTLY_AVAILABLE:
        _display_predictions_plotly(
            display_names, confidences, title, height, color_scale
        )
    elif MATPLOTLIB_AVAILABLE:
        _display_predictions_matplotlib(
            display_names, confidences, title, height, color_scale
        )
    else:
        # Fallback to simple text display
        _display_predictions_text(classes, confidences, title)


def _display_predictions_plotly(
    classes: List[str],
    confidences: List[float],
    title: str,
    height: int,
    color_scale: str
) -> None:
    """Display predictions using Plotly (internal function)."""
    
    # Create color gradient
    colors = px.colors.sample_colorscale(
        color_scale, [i / len(confidences) for i in range(len(confidences))]
    )
    
    # Create horizontal bar chart
    fig = go.Figure(data=[
        go.Bar(
            y=classes,
            x=confidences,
            orientation='h',
            marker=dict(
                color=colors,
                line=dict(color='white', width=1)
            ),
            text=[f'{conf:.1f}%' for conf in confidences],
            textposition='outside',
            hovertemplate='<b>%{y}</b><br>Confidence: %{x:.2f}%<extra></extra>'
        )
    ])
    
    # Update layout
    fig.update_layout(
        title=dict(
            text=title,
            font=dict(size=20, color='#2c3e50'),
            x=0.5,
            xanchor='center'
        ),
        xaxis=dict(
            title='Confidence (%)',
            range=[0, 100],
            showgrid=True,
            gridcolor='rgba(200, 200, 200, 0.2)'
        ),
        yaxis=dict(
            title='',
            autorange='reversed'  # Top prediction at top
        ),
        height=height,
        margin=dict(l=20, r=100, t=60, b=40),
        plot_bgcolor='rgba(240, 240, 240, 0.3)',
        paper_bgcolor='white',
        font=dict(size=12, color='#2c3e50'),
        hovermode='y'
    )
    
    # Display
    st.plotly_chart(fig, use_container_width=True)


def _display_predictions_matplotlib(
    classes: List[str],
    confidences: List[float],
    title: str,
    height: int,
    color_scale: str
) -> None:
    """Display predictions using Matplotlib (internal function)."""
    
    fig, ax = plt.subplots(figsize=(10, height / 100))
    
    # Create color map
    cmap = plt.cm.get_cmap(color_scale)
    colors = [cmap(i / len(confidences)) for i in range(len(confidences))]
    
    # Create horizontal bar chart
    y_pos = np.arange(len(classes))
    bars = ax.barh(y_pos, confidences, color=colors, edgecolor='white', linewidth=1)
    
    # Add value labels
    for i, (bar, conf) in enumerate(zip(bars, confidences)):
        ax.text(
            conf + 2, i, f'{conf:.1f}%',
            va='center', ha='left', fontsize=10, fontweight='bold'
        )
    
    # Formatting
    ax.set_yticks(y_pos)
    ax.set_yticklabels(classes)
    ax.invert_yaxis()  # Top prediction at top
    ax.set_xlabel('Confidence (%)', fontsize=12, fontweight='bold')
    ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
    ax.set_xlim(0, 100)
    ax.grid(axis='x', alpha=0.3, linestyle='--')
    ax.set_facecolor('#f8f9fa')
    
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()


def _display_predictions_text(
    classes: List[str],
    confidences: List[float],
    title: str
) -> None:
    """Display predictions as text (fallback)."""
    st.subheader(title)
    
    for i, (cls, conf) in enumerate(zip(classes, confidences), 1):
        # Emoji indicator
        if conf > 70:
            emoji = "🔴"  # High confidence
        elif conf > 40:
            emoji = "🟡"  # Medium confidence
        else:
            emoji = "🔵"  # Low confidence
        
        st.markdown(f"{emoji} **{i}. {cls}**: {conf:.1f}%")


def format_prediction_table(
    predictions: List[Tuple[str, float]]
) -> str:
    """
    Format predictions as HTML table.
    
    Args:
        predictions: List of (class_name, confidence) tuples
    
    Returns:
        HTML string for table
        
    Example:
        >>> preds = [('Healthy', 0.95), ('Disease', 0.05)]
        >>> html = format_prediction_table(preds)
        >>> st.markdown(html, unsafe_allow_html=True)
    """
    html = """
    <style>
    .pred-table {
        width: 100%;
        border-collapse: collapse;
        margin: 20px 0;
    }
    .pred-table th {
        background-color: #2c3e50;
        color: white;
        padding: 12px;
        text-align: left;
    }
    .pred-table td {
        padding: 10px;
        border-bottom: 1px solid #ddd;
    }
    .pred-table tr:hover {
        background-color: #f5f5f5;
    }
    .conf-bar {
        background: linear-gradient(to right, #3498db, #2ecc71);
        height: 20px;
        border-radius: 10px;
        display: inline-block;
    }
    </style>
    <table class="pred-table">
    <thead>
        <tr>
            <th>Rank</th>
            <th>Disease/Condition</th>
            <th>Confidence</th>
            <th>Visual</th>
        </tr>
    </thead>
    <tbody>
    """
    
    for i, (cls, conf) in enumerate(predictions, 1):
        percentage = conf * 100
        bar_width = int(percentage * 2)  # Scale to 200px max
        
        html += f"""
        <tr>
            <td><strong>{i}</strong></td>
            <td>{cls.replace('_', ' ').title()}</td>
            <td><strong>{percentage:.1f}%</strong></td>
            <td><div class="conf-bar" style="width: {bar_width}px;"></div></td>
        </tr>
        """
    
    html += """
    </tbody>
    </table>
    """
    
    return html


# ============================================================================
# GradCAM Visualization
# ============================================================================

def show_gradcam_overlay(
    original: Union[Image.Image, np.ndarray],
    heatmap: Union[Image.Image, np.ndarray],
    column_width: Optional[Tuple[int, int]] = None,
    titles: Optional[Tuple[str, str]] = None,
    captions: Optional[Tuple[str, str]] = None
) -> None:
    """
    Display original image and GradCAM heatmap side-by-side in Streamlit columns.
    
    Args:
        original: Original image (PIL or numpy array)
        heatmap: GradCAM heatmap overlay (PIL or numpy array)
        column_width: Tuple of (left_col, right_col) widths (default: (1, 1))
        titles: Tuple of (original_title, heatmap_title) (optional)
        captions: Tuple of (original_caption, heatmap_caption) (optional)
    
    Example:
        >>> from utils.gradcam import generate_gradcam_visualization
        >>> original = Image.open('leaf.jpg')
        >>> overlay = generate_gradcam_visualization(model, 'leaf.jpg', ...)
        >>> show_gradcam_overlay(original, overlay)
    """
    # Convert to PIL if numpy
    if isinstance(original, np.ndarray):
        original = Image.fromarray(original.astype('uint8'))
    if isinstance(heatmap, np.ndarray):
        heatmap = Image.fromarray(heatmap.astype('uint8'))
    
    # Default column widths
    if column_width is None:
        column_width = (1, 1)
    
    # Default titles
    if titles is None:
        titles = ("📷 Original Image", "🔥 GradCAM Heatmap")
    
    # Create columns
    col1, col2 = st.columns(column_width)
    
    with col1:
        st.markdown(f"### {titles[0]}")
        st.image(original, use_container_width=True)
        if captions and captions[0]:
            st.caption(captions[0])
    
    with col2:
        st.markdown(f"### {titles[1]}")
        st.image(heatmap, use_container_width=True)
        if captions and captions[1]:
            st.caption(captions[1])


def show_gradcam_grid(
    images: List[Union[Image.Image, np.ndarray]],
    titles: Optional[List[str]] = None,
    columns: int = 3
) -> None:
    """
    Display multiple GradCAM visualizations in a grid layout.
    
    Args:
        images: List of images (PIL or numpy arrays)
        titles: Optional titles for each image
        columns: Number of columns in grid
    
    Example:
        >>> overlays = [overlay1, overlay2, overlay3, overlay4]
        >>> show_gradcam_grid(overlays, columns=2)
    """
    if not images:
        st.warning("⚠️ No images to display")
        return
    
    # Convert all to PIL
    pil_images = []
    for img in images:
        if isinstance(img, np.ndarray):
            pil_images.append(Image.fromarray(img.astype('uint8')))
        else:
            pil_images.append(img)
    
    # Default titles
    if titles is None:
        titles = [f"Image {i+1}" for i in range(len(pil_images))]
    
    # Create grid
    rows = (len(pil_images) + columns - 1) // columns
    
    for row in range(rows):
        cols = st.columns(columns)
        for col_idx in range(columns):
            img_idx = row * columns + col_idx
            if img_idx < len(pil_images):
                with cols[col_idx]:
                    st.markdown(f"**{titles[img_idx]}**")
                    st.image(pil_images[img_idx], use_container_width=True)


def show_gradcam_comparison(
    original: Union[Image.Image, np.ndarray],
    overlays: List[Union[Image.Image, np.ndarray]],
    predictions: List[Tuple[str, float]]
) -> None:
    """
    Show original image with multiple GradCAM overlays for top predictions.
    
    Args:
        original: Original image
        overlays: List of GradCAM overlays (one per prediction)
        predictions: List of (class_name, confidence) tuples
    
    Example:
        >>> # Compare top-3 predictions
        >>> show_gradcam_comparison(original, [overlay1, overlay2, overlay3], predictions)
    """
    st.markdown("### 🔍 Model Focus Comparison")
    st.caption("See which regions the model focused on for each prediction")
    
    # Create columns: original + overlays
    num_cols = len(overlays) + 1
    cols = st.columns(num_cols)
    
    # Original
    with cols[0]:
        st.markdown("**📷 Original**")
        if isinstance(original, np.ndarray):
            original = Image.fromarray(original.astype('uint8'))
        st.image(original, use_container_width=True)
        st.caption("Input image")
    
    # Overlays
    for i, (overlay, (cls, conf)) in enumerate(zip(overlays, predictions), 1):
        with cols[i]:
            rank_emoji = ["🥇", "🥈", "🥉"][i-1] if i <= 3 else f"#{i}"
            st.markdown(f"**{rank_emoji} {cls.replace('_', ' ').title()}**")
            if isinstance(overlay, np.ndarray):
                overlay = Image.fromarray(overlay.astype('uint8'))
            st.image(overlay, use_container_width=True)
            st.caption(f"Confidence: {conf*100:.1f}%")


# ============================================================================
# Utility Functions
# ============================================================================

def create_confidence_indicator(
    confidence: float,
    thresholds: Tuple[float, float] = (0.7, 0.4)
) -> str:
    """
    Create HTML confidence indicator with color coding.
    
    Args:
        confidence: Confidence score (0-1)
        thresholds: (high, medium) thresholds
    
    Returns:
        HTML string with colored badge
        
    Example:
        >>> html = create_confidence_indicator(0.85)
        >>> st.markdown(html, unsafe_allow_html=True)
    """
    high_thresh, med_thresh = thresholds
    
    if confidence >= high_thresh:
        color = "#27ae60"  # Green
        label = "High Confidence"
        emoji = "✅"
    elif confidence >= med_thresh:
        color = "#f39c12"  # Orange
        label = "Medium Confidence"
        emoji = "⚠️"
    else:
        color = "#e74c3c"  # Red
        label = "Low Confidence"
        emoji = "❌"
    
    percentage = confidence * 100
    
    html = f"""
    <div style="
        background: linear-gradient(135deg, {color}22, {color}44);
        border-left: 4px solid {color};
        padding: 12px 16px;
        border-radius: 8px;
        margin: 10px 0;
    ">
        <span style="font-size: 24px;">{emoji}</span>
        <strong style="color: {color}; font-size: 18px; margin-left: 8px;">
            {label}
        </strong>
        <div style="font-size: 28px; color: {color}; font-weight: bold; margin-top: 8px;">
            {percentage:.1f}%
        </div>
    </div>
    """
    
    return html


def save_prediction_history(
    image: Image.Image,
    predictions: List[Tuple[str, float]],
    metadata: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Create history entry for prediction.
    
    Args:
        image: Input image
        predictions: List of (class_name, confidence) tuples
        metadata: Optional additional metadata (crop type, weather, etc.)
    
    Returns:
        Dictionary with history entry
        
    Example:
        >>> entry = save_prediction_history(image, predictions, {'crop': 'Potato'})
        >>> st.session_state['history'].append(entry)
    """
    from datetime import datetime
    
    entry = {
        'timestamp': datetime.now().isoformat(),
        'predictions': predictions,
        'image_size': image.size,
        'image_mode': image.mode,
    }
    
    if metadata:
        entry['metadata'] = metadata
    
    return entry


def get_crop_emoji(crop_name: str) -> str:
    """
    Get emoji for crop type.
    
    Args:
        crop_name: Name of crop
    
    Returns:
        Emoji string
        
    Example:
        >>> emoji = get_crop_emoji('Potato')
        >>> print(emoji)  # 🥔
    """
    crop_emojis = {
        'potato': '🥔',
        'tomato': '🍅',
        'wheat': '🌾',
        'rice': '🌾',
        'corn': '🌽',
        'sugarcane': '🎋',
        'cotton': '☁️',
        'apple': '🍎',
        'grape': '🍇',
        'orange': '🍊',
        'banana': '🍌',
    }
    
    crop_lower = crop_name.lower()
    for key, emoji in crop_emojis.items():
        if key in crop_lower:
            return emoji
    
    return '🌱'  # Default plant emoji


# ============================================================================
# Export List
# ============================================================================

__all__ = [
    # Class mapping
    'load_class_names',
    'load_class_mapping',
    
    # Image handling
    'bytesio_to_pil',
    'uploaded_file_to_pil',
    'pil_to_bytes',
    'resize_image',
    
    # Prediction visualization
    'display_predictions',
    'format_prediction_table',
    
    # GradCAM visualization
    'show_gradcam_overlay',
    'show_gradcam_grid',
    'show_gradcam_comparison',
    
    # Utilities
    'create_confidence_indicator',
    'save_prediction_history',
    'get_crop_emoji',
]
